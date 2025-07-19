from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import out  # Your prediction script

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as file:
                self.wfile.write(file.read())
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = urllib.parse.parse_qs(post_data)
            date = params.get('date', [''])[0]
            
            try:
                # Call your prediction function
                forecast = out.predict_weather(date)
                
                # Convert DataFrame to dictionary for JSON response
                forecast_dict = forecast.copy()
                forecast_dict['date'] = forecast_dict['date'].dt.strftime('%Y-%m-%d').tolist()
                result = {
                    'success': True,
                    'forecast': []
                }
                
                for i in range(len(forecast_dict['date'])):
                    day = {
                        'date': forecast_dict['date'][i]
                    }
                    for col in forecast.columns:
                        if col != 'date':
                            day[col] = float(forecast_dict[col][i])
                    result['forecast'].append(day)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

# Run server
httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
print("Server started at http://localhost:8000")
httpd.serve_forever()