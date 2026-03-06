from core.ollama import Ollama
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(self, ollama_service: Ollama, clients: dict[str, MCPClient]):
        self.ollama_service: Ollama = ollama_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(self, query: str) -> str:
        final_text_response = ""
        await self._process_query(query)

        while True:
            response = self.ollama_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            self.ollama_service.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                print(self.ollama_service.text_from_message(response))
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )
                self.ollama_service.add_user_message(
                    self.messages, tool_result_parts
                )
            else:
                final_text_response = self.ollama_service.text_from_message(
                    response
                )
                break

        return final_text_response
