#!/usr/bin/env python3
"""
Repository File Combiner - Web Interface
Combines all files from a GitHub repository into a single file and serves it as a download.
Run on port 8080, accessible from any IP (0.0.0.0)
"""

import os
import sys
import tempfile
import zipfile
import requests
import base64
from flask import Flask, render_template_string, request, send_file, flash, redirect
import io

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# HTML Template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Repository File Combiner</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #4CAF50;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .info {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 3px;
        }
        .error {
            background-color: #ffebee;
            border-left: 6px solid #f44336;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 3px;
        }
        .example {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Repository File Combiner</h1>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="info">
            <strong>How it works:</strong> Enter a GitHub repository URL or in the format "username/repository" 
            to combine all its files into a single downloadable file.
        </div>
        
        <form method="POST">
            <div class="form-group">
                <label for="repo">Repository:</label>
                <input type="text" id="repo" name="repo" 
                       placeholder="e.g., octocat/Hello-World or https://github.com/octocat/Hello-World" 
                       value="{{ request.form.repo if request.form.repo else '' }}" required>
                <div class="example">Examples: flask/flask, https://github.com/torvalds/linux</div>
            </div>
            
            <button type="submit" class="btn">Combine and Download</button>
        </form>
        
        <div class="footer">
            Server running on port 8080 • Accessible from any IP
        </div>
    </div>
</body>
</html>
'''

def parse_repo_input(repo_input):
    """Parse repository input to extract owner and repo name"""
    # Remove any trailing slashes
    repo_input = repo_input.rstrip('/')
    
    # Handle full GitHub URLs
    if 'github.com' in repo_input:
        # Extract owner/repo from URL
        parts = repo_input.split('github.com/')[-1].split('/')
        if len(parts) >= 2:
            return parts[0], parts[1].replace('.git', '')
    
    # Handle "owner/repo" format
    elif '/' in repo_input:
        parts = repo_input.split('/')
        if len(parts) == 2:
            return parts[0], parts[1].replace('.git', '')
    
    return None, None

def get_repo_contents(owner, repo, path=''):
    """Recursively get all files from a GitHub repository"""
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        contents = response.json()
        
        files = []
        
        for item in contents:
            if item['type'] == 'file':
                # Get file content
                file_response = requests.get(item['download_url'])
                file_response.raise_for_status()
                files.append({
                    'path': item['path'],
                    'content': file_response.text
                })
            elif item['type'] == 'dir':
                # Recursively get files from subdirectory
                files.extend(get_repo_contents(owner, repo, item['path']))
        
        return files
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error accessing repository: {str(e)}")

def create_combined_file(files):
    """Create a single file combining all repository files"""
    output = io.BytesIO()
    
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add a README first
        readme_content = f"""Repository File Combination
Generated on: Combined all files from the repository
Total files combined: {len(files)}

This archive contains all files from the repository.
"""
        zipf.writestr('00_COMBINED_README.txt', readme_content)
        
        # Add all files
        for file_info in files:
            zipf.writestr(file_info['path'], file_info['content'])
        
        # Also create a single combined text file with all contents
        combined_content = ""
        for file_info in sorted(files, key=lambda x: x['path']):
            combined_content += f"\n{'='*80}\n"
            combined_content += f"FILE: {file_info['path']}\n"
            combined_content += f"{'='*80}\n\n"
            combined_content += file_info['content']
            combined_content += "\n\n"
        
        zipf.writestr('00_ALL_FILES_COMBINED.txt', combined_content)
    
    output.seek(0)
    return output

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        repo_input = request.form.get('repo', '').strip()
        
        if not repo_input:
            flash('Please enter a repository', 'error')
            return redirect('/')
        
        # Parse repository input
        owner, repo = parse_repo_input(repo_input)
        
        if not owner or not repo:
            flash('Invalid repository format. Please use "owner/repo" or a GitHub URL', 'error')
            return redirect('/')
        
        try:
            # Get all files from repository
            flash(f'Fetching files from {owner}/{repo}...', 'info')
            files = get_repo_contents(owner, repo)
            
            if not files:
                flash('No files found in repository', 'error')
                return redirect('/')
            
            # Create combined file
            flash(f'Found {len(files)} files. Creating download...', 'info')
            combined_file = create_combined_file(files)
            
            # Send file for download
            return send_file(
                combined_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'{repo}_combined_files.zip'
            )
            
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return redirect('/')
    
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("="*50)
    print("Repository File Combiner Server")
    print("="*50)
    print("Server starting on http://0.0.0.0:8080")
    print("Access from any device on your network using your IP address")
    print("\nPress Ctrl+C to stop the server")
    print("="*50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)