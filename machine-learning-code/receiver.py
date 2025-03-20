import socket

# Set up the server
host = '127.0.0.1'  # Localhost
port = 65432  # Port number

# Start the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print(f"Listening on {host}:{port}")

# Accept a client connection
client_socket, address = server_socket.accept()
print(f"Connected to {address}")

# Receive data from the client
try:
    while True:
        # Receive 1024 bytes of data
        data = client_socket.recv(1024)

        if not data:
            break

        try:
            # Try to decode the data using UTF-8
            decoded_data = data.decode('utf-8')
            print(f"Received (UTF-8): {decoded_data}")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, fallback to a different encoding or handle binary data
            print("Error decoding UTF-8, handling raw data")
            decoded_data = data.decode('latin-1')  # Try another encoding like 'latin-1'
            print(f"Received (Fallback): {decoded_data}")

except Exception as e:
    print(f"Error: {e}")
finally:
    # Close the client and server sockets
    client_socket.close()
    server_socket.close()

