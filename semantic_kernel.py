import os
import asyncio
from pathlib import Path

from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.functions import kernel_function
from typing import Annotated
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from dotenv import find_dotenv, load_dotenv



async def main():
    os.system('cls' if os.name=='nt' else 'clear')

    script_dir = Path(__file__).parent
    file_path = script_dir / 'data.txt'
    with file_path.open('r') as file:
        data = file.read() + "\n"

    user_prompt = input(f"Here is the expenses data in your file:\n\n{data}\n\nWhat would you like me to do with it?\n\n")

    await process_expenses_data (user_prompt, data)

async def process_expenses_data(prompt, expenses_data):
    load_dotenv()
    ai_agent_settings = AzureAIAgentSettings()

    async with (
        DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True) as creds,
        AzureAIAgent.create_client(
            credential=creds
        ) as project_client
    ):
        
        expenses_agent_def = await project_client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="expenses_agent",
            instructions="""You are an AI assistant for expense claim submission.
                            When a user submits expenses data and requests a expense claim, use the plug-in function to send an email to neil.prash0714@gmail.com with the subject 'Expense Claim' and a body that contains the itemized expenses with a total.
                            Then confirm to the user that you've done so.
                            If the user asks for a P&L statement, use the plug-in function to send an email to neil.prash0714@gmail.com with the subject 'P&L Statement' and a body that contains a P&L statement containing debits and credits.
                            Then confirm to the user that you've done so.
                            If the user asks for anything except a claim, say 'I cannot help with that'"""
        )

        expenses_agent = AzureAIAgent(
            client=project_client,
            definition=expenses_agent_def,
            plugins=[EmailPlugin()]
        )

        thread: AzureAIAgentThread = AzureAIAgentThread(client=project_client)
        try:
            prompt_messages = [f"{prompt}: {expenses_data}"]
            response = await expenses_agent.get_response(thread_id=thread.id, messages=prompt_messages)
            print(f"\n# {response.name}:\n{response}")
        except Exception as e:
            print(e)
        finally:
            await thread.delete() if thread else None
            await project_client.agents.delete_agent(expenses_agent.id)

class EmailPlugin:
    @kernel_function(description="Sends an email.")
    def send_email(self,
                   to: Annotated[str, "Who to send the email to"],
                   subject: Annotated[str, "The subject of the email"],
                   body: Annotated[str, "The text body of the email"]):
        print("\nTo: ", to)
        print("Subject: ", subject)
        print(body, "\n")


if __name__ == "__main__":
    asyncio.run(main())
