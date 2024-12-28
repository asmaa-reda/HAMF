import os
import requests
import pandas as pd
from imapclient import IMAPClient

def collect_from_web(url):
    response = requests.get(url)
    return response.text

def collect_from_api(api_url, headers):
    response = requests.get(api_url, headers=headers)
    return response.json()

def collect_from_email(server, username, password):
    with IMAPClient(server) as client:
        client.login(username, password)
        client.select_folder('INBOX')
        messages = client.search(['NOT DELETED'])
        return client.fetch(messages, ['BODY[]'])

def collect_from_csv(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Example: Collect data from multiple sources
    web_data = collect_from_web("https://example.com")
    api_data = collect_from_api("https://api.example.com", headers={"Authorization": "Bearer token"})
    email_data = collect_from_email("imap.example.com", "user@example.com", "password")
    csv_data = collect_from_csv("./data/input.csv")

    # Save raw data
    with open("./repos/data/raw_web.txt", "w") as f:
        f.write(web_data)

    pd.DataFrame(api_data).to_csv("./data/raw_api.csv", index=False)
    pd.DataFrame(email_data).to_csv("./data/raw_email.csv", index=False)
    csv_data.to_csv("./data/raw_csv.csv", index=False)
