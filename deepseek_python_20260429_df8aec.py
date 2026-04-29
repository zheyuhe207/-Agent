"""
智能工单根因分析与自动修复 Agent
基于 LangGraph 的多智能体协作系统

依赖安装：
pip install langgraph langchain-openai python-dotenv requests

环境变量（如需真实 OpenAI）：
export OPENAI_API_KEY=your_key
或者设置 USE_MOCK=true 使用模拟 LLM（无需 API Key）
"""

import json
import re
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests  # 用于真实 Kibana API
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# ==================== 配置 ====================
USE_MOCK = True  # 设为 False 时使用真实 OpenAI（需配置 API Key）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
KIBANA_BASE_URL = os.getenv("KIBANA_BASE_URL", "http://localhost:5601")  # 替换为实际地址
KIBANA_INDEX = "logs-*"

# 模拟 Runbook 知识库
RUNBOOK_DB = {
    "OOM": "内存溢出，建议：1. 增加 Pod memory limit；2. 检查代码内存泄漏；3. 重启 Pod。",
    "ConnectionTimeout": "依赖服务超时，建议：1. 检查下游服务健康状态；2. 增加超时配置；3. 重试机制。",
    "DiskFull": "磁盘空间不足，建议：1. 清理日志文件；2. 扩容 PVC；3. 调整日志轮转策略。",
    "CPUPressure": "CPU 节流严重，建议：1. 增加 CPU request/limit；2. 优化代码性能；3. 水平扩容。"
}

# ==================== 状态定义 ====================
class AgentState(TypedDict):
    raw_alert: Dict[str, Any]                     # 原始告警
    ticket: Optional[Dict[str, Any]]              # 工单结构化信息
    logs: Optional[List[str]]                     # 检索到的日志
    root_cause: Optional[str]                     # 推理出的根因
    runbook_entry: Optional[str]                  # 匹配的 Runbook 建议
    fix_command: Optional[str]                    # 生成的修复命令
    risk_analysis: Optional[str]                  # 风险分析
    auto_fix_triggered: bool                      # 是否触发自动修复
    final_report: Optional[str]                   # 最终报告

# ==================== 模拟 LLM（无需 API Key）====================
class MockLLM:
    def invoke(self, messages):
        # 模拟 LLM 响应，根据消息内容返回合理的推理结果
        user_text = messages[-1].content if messages else ""
        if "提取工单信息" in user_text:
            return MockLLMResponse("工单信息：服务 user-service，错误码 137，时间 2025-04-01T10:00:00Z，Pod user-pod-7d8f9")
        elif "推理根因" in user_text or "日志" in user_text:
            return MockLLMResponse("根因分析：错误码 137 表示 OOM（内存溢出）。日志中出现 'Out of memory: Kill process' 和内存持续增长模式，判定为内存泄漏导致 OOM Kill。")
        elif "查询 Runbook" in user_text:
            return MockLLMResponse("匹配 Runbook：OOM。建议增加 memory limit 到 2Gi，并代码排查内存泄漏。")
        elif "生成修复命令" in user_text:
            return MockLLMResponse("kubectl patch deployment user-service -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"app\",\"resources\":{\"limits\":{\"memory\":\"2Gi\"}}}]}}}}'")
        elif "风险分析" in user_text:
            return MockLLMResponse("风险：低。变更仅限于资源 limit，不会影响服务可用性。需观察重启后内存趋势。")
        else:
            return MockLLMResponse("模拟响应完成。")

class MockLLMResponse:
    def __init__(self, content):
        self.content = content

# 根据配置选择真实 LLM 或模拟
def get_llm():
    if USE_MOCK:
        return MockLLM()
    else:
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

llm = get_llm()

# ==================== 工具函数 ====================
def extract_alert_info_prompt(alert: Dict[str, Any]) -> str:
    """生成工单提取的 prompt"""
    return f"""
请从以下告警 JSON 中提取工单核心信息：提取时间戳、服务名、错误码、Pod 名称（如果有）。
输出格式：一句话描述。
告警内容：{json.dumps(alert, indent=2)}
"""

def query_kibana_logs(service: str, time_range_minutes: int = 15, max_entries: int = 50) -> List[str]:
    """真实查询 Kibana API（需要配置实际 endpoint 和认证）"""
    if USE_MOCK:
        # 模拟日志数据
        current_time = datetime.utcnow()
        logs = []
        for i in range(10):
            log_time = current_time - timedelta(minutes=i)
            logs.append(f"{log_time.isoformat()}Z [ERROR] service={service} - Out of memory: Kill process or sacrifice child")
        logs.append(f"{current_time.isoformat()}Z [FATAL] service={service} - Killed process due to memory pressure")
        return logs

    # 真实 Kibana 查询（示例，需根据实际 API 调整）
    # 参考: https://www.elastic.co/guide/en/kibana/current/api.html
    # 此处仅为骨架
    headers = {"kbn-xsrf": "true"}
    # 需要认证的话加上 Authorization
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"kubernetes.container_name": service}},
                    {"range": {"@timestamp": {"gte": f"now-{time_range_minutes}m"}}}
                ]
            }
        },
        "size": max_entries,
        "sort": [{"@timestamp": "desc"}]
    }
    try:
        resp = requests.post(
            f"{KIBANA_BASE_URL}/api/console/proxy?path=/{KIBANA_INDEX}/_search&method=GET",
            json=query,
            headers=headers,
            timeout=10
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        logs = [hit["_source"].get("message", "") for hit in hits]
        return logs
    except Exception as e:
        print(f"Kibana 查询失败: {e}")
        return []

def root_cause_analysis_prompt(logs: List[str]) -> str:
    """生成根因分析的 prompt"""
    log_sample = "\n".join(logs[:30]) if logs else "无日志"
    return f"""
请根据以下日志片段，推理故障根因。使用思维链：先看错误码/异常类型，再看时间序列模式，最后给出结论。
日志：
{log_sample}
输出根因描述（例如：OOM、连接超时、磁盘满等）。
"""

def lookup_runbook_prompt(root_cause: str) -> str:
    """查询 Runbook 知识库（可结合向量检索，此处用关键词匹配+LLM）"""
    # 先尝试关键词匹配
    for key in RUNBOOK_DB:
        if key.lower() in root_cause.lower():
            return f"匹配到 Runbook 条目：{key}\n建议：{RUNBOOK_DB[key]}"

    # 否则让 LLM 从知识库中找最相近的
    return f"""
已知 Runbook 知识库：
{json.dumps(RUNBOOK_DB, indent=2)}
根因：{root_cause}
请返回最匹配的 Runbook 建议。
"""

def generate_fix_command_prompt(runbook_entry: str, root_cause: str) -> str:
    """生成具体的 kubectl 或配置命令"""
    return f"""
根据根因 ({root_cause}) 和 Runbook 建议 ({runbook_entry})，生成一条可执行的 kubectl 或配置修改命令。
例如：kubectl patch deployment ... 或 helm upgrade ...
只输出命令本身。
"""

def risk_analysis_prompt(fix_command: str, runbook_entry: str) -> str:
    """分析修复命令的风险"""
    return f"""
修复命令：{fix_command}
Runbook 建议：{runbook_entry}
请评估风险等级（高/中/低）和理由。输出一句话。
"""

def generate_final_report(state: AgentState) -> str:
    """生成最终报告"""
    report = f"""
=== 智能工单根因分析报告 ===
告警时间：{state['ticket'].get('timestamp', '未知')}
服务名：{state['ticket'].get('service', '未知')}
错误码：{state['ticket'].get('error_code', '未知')}
根因：{state['root_cause']}
Runbook 建议：{state['runbook_entry']}
修复命令：{state['fix_command']}
风险分析：{state['risk_analysis']}
自动修复触发：{state['auto_fix_triggered']}
"""
    return report

# ==================== Agent 节点定义 ====================
def ticket_agent(state: AgentState) -> AgentState:
    """工单 Agent：解析告警，输出结构化工单"""
    alert = state["raw_alert"]
    prompt = extract_alert_info_prompt(alert)
    response = llm.invoke([HumanMessage(content=prompt)])
    # 简单解析响应（实际可更结构化）
    text = response.content
    # 用正则或关键词提取（demo 简化）
    ticket = {
        "timestamp": re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', text).group(0) if re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', text) else "unknown",
        "service": re.search(r'service (\S+)', text).group(1) if re.search(r'service (\S+)', text) else "unknown",
        "error_code": re.search(r'错误码 (\d+)', text).group(1) if re.search(r'错误码 (\d+)', text) else "137",
        "pod": re.search(r'pod (\S+)', text).group(1) if re.search(r'pod (\S+)', text) else "unknown"
    }
    state["ticket"] = ticket
    return state

def log_agent(state: AgentState) -> AgentState:
    """日志侦探 Agent：检索 Kibana 日志"""
    service = state["ticket"].get("service", "unknown")
    logs = query_kibana_logs(service, time_range_minutes=15)
    state["logs"] = logs
    return state

def root_cause_agent(state: AgentState) -> AgentState:
    """根因推理 Agent：使用思维链分析日志"""
    logs = state.get("logs", [])
    prompt = root_cause_analysis_prompt(logs)
    response = llm.invoke([HumanMessage(content=prompt)])
    state["root_cause"] = response.content
    return state

def runbook_agent(state: AgentState) -> AgentState:
    """修复建议 Agent：查询 Runbook 知识库"""
    root_cause = state["root_cause"]
    prompt = lookup_runbook_prompt(root_cause)
    response = llm.invoke([HumanMessage(content=prompt)])
    state["runbook_entry"] = response.content
    return state

def fix_command_agent(state: AgentState) -> AgentState:
    """生成修复指令 Agent"""
    prompt = generate_fix_command_prompt(state["runbook_entry"], state["root_cause"])
    response = llm.invoke([HumanMessage(content=prompt)])
    state["fix_command"] = response.content
    return state

def risk_agent(state: AgentState) -> AgentState:
    """风险分析 Agent"""
    prompt = risk_analysis_prompt(state["fix_command"], state["runbook_entry"])
    response = llm.invoke([HumanMessage(content=prompt)])
    state["risk_analysis"] = response.content
    return state

def auto_fix_decision(state: AgentState) -> AgentState:
    """自动修复决策：根据根因和风险决定是否自动执行"""
    root_cause = state["root_cause"]
    risk = state["risk_analysis"].lower()
    # 简单策略：如果根因包含 OOM 且风险等级低，则触发自动修复
    if "oom" in root_cause.lower() and "低" in risk:
        state["auto_fix_triggered"] = True
        # 模拟执行修复命令（实际可调用 kubectl）
        print(f"[自动修复] 执行命令: {state['fix_command']}")
        # 这里可添加实际执行代码，如 subprocess.run(...)
    else:
        state["auto_fix_triggered"] = False
    return state

def report_agent(state: AgentState) -> AgentState:
    """生成最终报告"""
    state["final_report"] = generate_final_report(state)
    return state

# ==================== 构建 LangGraph 工作流 ====================
def build_agent_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("ticket", ticket_agent)
    workflow.add_node("logs", log_agent)
    workflow.add_node("root_cause", root_cause_agent)
    workflow.add_node("runbook", runbook_agent)
    workflow.add_node("fix_cmd", fix_command_agent)
    workflow.add_node("risk", risk_agent)
    workflow.add_node("auto_fix", auto_fix_decision)
    workflow.add_node("report", report_agent)

    # 设置边
    workflow.set_entry_point("ticket")
    workflow.add_edge("ticket", "logs")
    workflow.add_edge("logs", "root_cause")
    workflow.add_edge("root_cause", "runbook")
    workflow.add_edge("runbook", "fix_cmd")
    workflow.add_edge("fix_cmd", "risk")
    workflow.add_edge("risk", "auto_fix")
    workflow.add_edge("auto_fix", "report")
    workflow.add_edge("report", END)

    return workflow.compile()

# ==================== 示例运行 ====================
if __name__ == "__main__":
    # 模拟一条告警
    sample_alert = {
        "alertname": "PodMemoryUsageHigh",
        "labels": {
            "service": "user-service",
            "pod": "user-pod-7d8f9",
            "namespace": "production"
        },
        "annotations": {
            "summary": "Memory usage above 95%",
            "error_code": "137"
        },
        "startsAt": "2025-04-01T10:00:00Z"
    }

    initial_state: AgentState = {
        "raw_alert": sample_alert,
        "ticket": None,
        "logs": None,
        "root_cause": None,
        "runbook_entry": None,
        "fix_command": None,
        "risk_analysis": None,
        "auto_fix_triggered": False,
        "final_report": None
    }

    graph = build_agent_graph()
    final_state = graph.invoke(initial_state)

    print("\n" + "="*50)
    print("最终报告：")
    print(final_state["final_report"])
    print("="*50)