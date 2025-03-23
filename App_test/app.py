from time import sleep
import os
import webview
import json

html_file = os.path.abspath("load_session.html")
session_path = os.path.abspath("sessions/")

class Api:
    def get_value(self):
        return "Value from Python"

    def get_sessions(self):
        session_files = os.listdir(session_path)
        return session_files

    def get_filedata(self, filename):
        with open(f'{session_path}/{filename}', 'r') as f:
            data = json.load(f)
            return data
        
        return {}

    def log(self, text):
        print('log: ',text)



api = Api()

def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()


def load_html(window):
    html = read_file(html_file)
    window.load_html(html)


if __name__ == '__main__':
    window = webview.create_window('Load HTML Example', f'file://{html_file}', width=1200, height=750, js_api=api)
    # window = webview.create_window('Load HTML Example', f'file://{html_file}', fullscreen=True, js_api=api)
    webview.start()