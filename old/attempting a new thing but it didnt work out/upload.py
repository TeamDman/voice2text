import requests

def upload_file(api_url, file_path, api_key=None):
    # Prepare the headers dictionary
    headers = {}
    if api_key:
        headers['Authorization'] = api_key

    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        # Prepare the files dictionary
        files = {'file': (file_path, f, 'audio/wav')}
        # Make the POST request to upload the file
        response = requests.post(api_url, files=files, headers=headers)
    
    return response

if __name__ == "__main__":
    # Define the API URL
    api_url = "http://localhost:8756/transcribe"

    # Define the path to your WAV file
    file_path = "untitled.wav"

    # Define your API key (optional, remove or adjust if not needed)
    api_key = "boy_i_sure_hope_you_supplied_one_from_the_command_line_cause_this_string_is_going_in_the_repo"
    
    # Call the function to upload the file
    response = upload_file(api_url, file_path, api_key)

    # Print the server's response
    print("Server responded with status code:", response.status_code)
    print("Response body:", response.text)
