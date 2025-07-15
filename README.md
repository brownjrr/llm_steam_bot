# llm_steam_bot
This project will use LLMs to create a chatbot capable of answering specific questions about Steam games.

---

## 1. Install Python 3.11.9

Ensure you have **Python 3.11.9** installed on your system.

Download it from the official Python website:  
https://www.python.org/downloads/

---

## 2. Configure AWS Credentials

To configure AWS access credentials, you will need the AWS CLI

You can install the CLI here:  
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

Once installed, configure your credentials using the following commands with your personal information

```bash
aws configure set default.aws_access_key_id {access_key_id}
aws configure set default.aws_secret_access_key {secret_access_key}
aws configure set default.aws_session_token {session_token}
aws configure set default.region us-east-1
```

---

## 3. Pip install the Requirements

we first recommend you use a virtual environment to seperate this projects requirements from any other ones that may already be installed
to do so run (using cmd)
```bash
python -m venv venv
```
Then activate the virtual environemnt by running (again using cmd)
```bash
.\venv\Scripts\activate
```

You can verify the environemnt is activte if you see (venv) before your prompts in cmd

Then you are ready to install the necessary packages by running
pip install the reqiured packages by running this in the project directory:
```bash
pip install -r requirements.txt
```
This may take a while if you do not already have these packages cached

---

## 4. Run the Dash Application

Navigate to the app folder by running
```bash
cd app
```
Then you can run the application by running app.py with this command
```bash
python app.py
```

This will start the Dash application on localhost port 8050
You can navigate to that in any browser by typing localhost:8050 or 127.0.0.1:8050

And that's it! You can now interact with the application and ask it questions about different games such as:
* "Tell me more about the reception of Left 4 Dead 2."

Or ask for some game recommendations with a prompt such as:
* Give me some recommendations for a game similar to Apex Legends