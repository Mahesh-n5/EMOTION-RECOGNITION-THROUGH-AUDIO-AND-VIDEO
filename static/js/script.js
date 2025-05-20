const video = document.getElementById('video');
const emotionDiv = document.getElementById('emotion');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;

        setInterval(() => {
            fetch('/predict', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                emotionDiv.textContent = `Emotion: ${data.emotion}`;
            });
        }, 1000); 
    })
    .catch((error) => {
        console.error("Error accessing webcam: ", error);
    });
