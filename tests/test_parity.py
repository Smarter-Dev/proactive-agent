from pydantic_ai.models.test import TestModel

from proactive_agent.parity import build_proactive_agent, parity_tool_functions

PARITY_TOOL_NAMES = {
    "web_search",
    "web_read",
    "list_available_reactions",
    "add_reaction",
    "report_behavior",
    "run_code",
    "generate_image",
    "remember",
    "register_handler",
    "list_handlers",
    "delete_handler",
}


def test_standalone_registers_the_production_chat_tool_surface():
    assert {tool.__name__ for tool in parity_tool_functions()} == PARITY_TOOL_NAMES
    agent = build_proactive_agent(
        TestModel(custom_output_text="done"), system_prompt="s"
    )
    assert PARITY_TOOL_NAMES <= set(agent._function_toolset.tools)
