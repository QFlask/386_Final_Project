from ollama import Client
from pydantic import BaseModel
from typing import Optional
import sys

LLM_MODEL: str = "gemma3:27b"  #  this is running on the AI server
client: Client = Client(host="http://ai.dfec.xyz:11434")  # this is the AI server


class User(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    hobbies: Optional[str] = None


# class UserList(BaseModel):
#   users: list[User]


def get_users(prompt: str) -> User:
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": "respond in JSON, with the fields name, age, hobbies (if provided)",
            },
            {"role": "user", "content": prompt},
        ],
        model=LLM_MODEL,
        format=User.model_json_schema(),
    )

    print(response.message.content)

    users = User.model_validate_json(response.message.content)

    return users


def main():
    # verify system arguments
    if len(sys.argv) != 2:
        print('Usage: python user_creation.py "prompt"')
        exit(1)

    prompt: str = sys.argv[1]
    print(f"Creating users from the prompt: \n{prompt}")

    users = get_users(prompt)

    print(f"Users created:\n{users}")


if __name__ == "__main__":
    main()
