from omegaconf import OmegaConf

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.llm.giga_chat import GigaChatLLM
from src.agent.tools.rag_vnd import vnd_tool
from src.agent.tools.reasoning import reasoning_tool
from src.agent.tools.web_search import web_search_tool
from src.agent.prompts import REACT_AGENT_SYSTEM

def create_agent(cfg_path = "configs/deployment_config.py"):
    cfg = OmegaConf.load(cfg_path)
    llm = GigaChatLLM(cfg['llm'])
    tools = {
        'vnd': vnd_tool,
        'reasoning': reasoning_tool,
        'web_search': web_search_tool
    }
    
    agent_tools = [tools[tool_name] for tool_name in tools if cfg['agent']['tools'][tool_name]]

    llm_with_tools = llm.bind_functions(agent_tools)
    agent = create_react_agent(llm_with_tools, 
                                agent_tools, 
                                checkpointer=MemorySaver(),
                                prompt=REACT_AGENT_SYSTEM)
    return agent
