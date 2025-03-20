using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System;

public class YogaSocketServer : MonoBehaviour
{
    // TCP Listener to accept client connections
    private TcpListener tcpListener;

    // TCP Client for the connected client
    private TcpClient connectedClient;

    // The transform of the GameObject this script is attached to
    private Transform m_Transform;

    // IP address and port the server will listen on
    private string host = "127.0.0.1";
    private int port = 65432;

    // Animator to control the yoga animations
    private Animator myAnimator;

    void Start()
    {
        myAnimator = this.GetComponent<Animator>();
        IPAddress ipAddress = IPAddress.Parse(host);
        tcpListener = new TcpListener(ipAddress, port);
        tcpListener.Start();
        Debug.Log($"Listening on {host}:{port}");
        m_Transform = gameObject.GetComponent<Transform>();

        // Start the coroutine to accept client connections
        StartCoroutine(AcceptClientCoroutine());
    }

    // Coroutine to accept client connections
    IEnumerator AcceptClientCoroutine()
    {
        // Loop indefinitely
        while (true)
        {
            // Accept a client connection if one is pending
            if (tcpListener.Pending())
            {
                connectedClient = tcpListener.AcceptTcpClient();
                Debug.Log("Connected by " + connectedClient.Client.RemoteEndPoint.ToString());

                // Start the coroutine to receive data from the connected client
                StartCoroutine(ReceiveDataCoroutine());
            }

            // Yield until the next frame
            yield return null;
        }
    }

    // Coroutine to receive data from the connected client
    IEnumerator ReceiveDataCoroutine()
    {
        // Get the network stream from the connected client
        NetworkStream networkStream = connectedClient.GetStream();

        // Loop indefinitely
        while (true)
        {
            // Read data from the network stream if available
            if (networkStream.DataAvailable)
            {
                byte[] buffer = new byte[1024];
                int bytesRead = networkStream.Read(buffer, 0, buffer.Length);
                string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                Debug.Log("Received: " + dataReceived);

                // Process the received data to trigger yoga animations
                try
		{
    			
    			int yogaPoseIndex = int.Parse(dataReceived.Trim());

        		// Update the Animator to trigger the corresponding yoga posture
        		myAnimator.SetInteger("Pose", yogaPoseIndex);
        		Debug.Log($"Playing Yoga Pose Animation {yogaPoseIndex}");
    			
		}
		catch (Exception ex)
		{
    			Debug.LogError("Error parsing received data: " + ex.Message);
		}

            }

            // Yield until the next frame
            yield return null;
        }
    }

    // Clean up when the application is closed
    void OnApplicationQuit()
    {
        // Stop the TCP Listener and close the connected client if they exist
        if (tcpListener != null)
        {
            tcpListener.Stop();
            tcpListener = null;
        }

        if (connectedClient != null)
        {
            connectedClient.Close();
            connectedClient = null;
        }
    }
}
