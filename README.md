### Note: Link to [Deriva's Charging Guidelines Excel Sheet](https://derivaenergy.sharepoint.com/:x:/r/sites/DerivaFinance/_layouts/15/Doc.aspx?sourcedoc=%7B3CD9F65D-C693-4CE8-904C-91074451F098%7D&file=Deriva%20OM%20Charging%20Guidelines.xlsx&wdOrigin=TEAMS-MAGLEV.p2p_ns.rwc&action=default&mobileredirect=true)

## Running Locally Simplified

1. Don't forget to get the correct credentials from aws.
2. Create a virtual environment and download the requirements. (Note: add --ignore-installed to it if there's an error in already installed packages)
3. `streamlit run app.py` or `py -m streamlit run app.py`.

## Running on EC2 Simplified
1. `git clone` this repo
2. `nohup streamlit run app.py --server.port 80 --server.address 0.0.0.0`
3. To stop it, `sudo lsof -i :80`, and then `sudo kill -9 [the PID]`

---

# A More Detailed Explaination

## How to Login EC2
1. Log in to the AWS Account: DER-PROD-GEN-AI-US-EAST-1
2. In "Recently visited", click on EC2. If EC2 is not in there, just search it in the search bar and click on EC2.
3. Look at the EC2 sidebar, and click on Instances.
4. As of August 2025, you will see 2 instances: ChatbotServerLinux and BatteryOptimizationServer. Click the checkbox for the ChatbotServerLinux, and click on the blue "Connect" button on top.
5. Set the Connection type to "Connect using a Private IP" and leave the rest to default. Then click the orange Connect button.
You should instantly be redirected to the Amazon Linux console. Then type in the command `sudo su -`

## How to Update the App (including if the repository is not in Jeremy's Github anymore)
0. You must always be in `sudo su -`
1. If you need to update the Chatbot, type in `rm -rf chatbot`, or rm -rf [the new folder's name]
2. Then type in git clone `https://github.com/deriva-jer/chatbot.git` or `git clone [the new github repository link]`
3. Then type in `cd chatbot` or `cd the new folder's name`.
4. Then type in `nohup streamlit run app.py --server.port 80 --server.address 0.0.0.0` to run the app in the server 24/7.

## How to Stop The Server (Do this before you remove the old Battery repository and before you clone a new one)
1. After doing `sudo su -`, type in `sudo lsof -i :80`
2. You should see `PID` and a number underneath (ex: 36891)
3. Then type in `sudo kill -9 36891` or `sudo kill -9 [PID number]`
4. This should instantly stop the server. And to run it again, just cd to your chatbot folder and type in `nohup streamlit run app.py --server.port 80 --server.address 0.0.0.0`

---

**Some more notes:**

For AWS inquiry: Sandeep

Finance: Gavin Humphrey

S3 - data \
Bedrock -> knowledge base - index \
DynamoDB -> memory/database \
Python script -> Streamlit app \
EC2 -> Instances - server - host the app

---

**Deployment videos**
- https://www.youtube.com/watch?v=oynd7Xv2i9Y
- https://www.youtube.com/watch?v=DflWqmppOAg

**Creating history langchain with dynamodb**
- https://python.langchain.com/v0.2/docs/integrations/memory/aws_dynamodb/
- https://aws.amazon.com/blogs/database/build-a-scalable-context-aware-chatbot-with-amazon-dynamodb-amazon-bedrock-and-langchain/
