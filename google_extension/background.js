// This is a service worker that runs in the background

// Listen for installation
chrome.runtime.onInstalled.addListener(function () {
  console.log("LaTeX Snapshot Selector installed");
});

// Listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "captureComplete") {
    console.log("Capture completed with data:", request.data);

    // Capture the screenshot of the selected area
    chrome.tabs.captureVisibleTab(null, { format: 'png' }, function (dataUrl) {
      const captureData = request.data;

      // Create a canvas to crop the screenshot
      const canvas = new OffscreenCanvas(captureData.width, captureData.height);
      const ctx = canvas.getContext('2d');

      // Convert dataUrl to bitmap for OffscreenCanvas
      fetch(dataUrl)
        .then(response => response.blob())
        .then(blob => createImageBitmap(blob))
        .then(imageBitmap => {
          // Draw only the selected portion of the screenshot
          ctx.drawImage(
            imageBitmap,
            captureData.x * captureData.devicePixelRatio,
            captureData.y * captureData.devicePixelRatio,
            captureData.width * captureData.devicePixelRatio,
            captureData.height * captureData.devicePixelRatio,
            0, 0, captureData.width, captureData.height
          );

          // Convert the canvas to a data URL
          return canvas.convertToBlob({ type: 'image/png' });
        })
        .then(blob => {
          // Convert blob to base64
          const reader = new FileReader();
          reader.readAsDataURL(blob);
          reader.onloadend = function () {
            const croppedDataUrl = reader.result;

            // Store the image for potential future use
            chrome.storage.local.set({
              capturedImage: croppedDataUrl
            }, function () {
              // Send to API (or mock for now)
              sendToApi(croppedDataUrl, sendResponse);
            });
          };
        })
        .catch(error => {
          console.error("Error capturing screenshot:", error);
          sendResponse({ status: "error", message: error.toString() });
        });

      return true; // Keep the message channel open for the async response
    });

    return true; // Keep the message channel open for the async response
  }

  // Handle API send request from the popup
  if (request.action === "sendToApi") {
    sendToApi(request.dataUrl, sendResponse);
    return true; // Keep the message channel open for the async response
  }
});

// Function to send image data to the LaTeX conversion API (now expects plain text response)
function sendToApi(dataUrl, sendResponse) {
  // Convert data URL to Blob
  const byteString = atob(dataUrl.split(',')[1]);
  const mimeString = dataUrl.split(',')[0].split(':')[1].split(';')[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  const blob = new Blob([ab], { type: mimeString });

  // Create FormData for the API request
  const formData = new FormData();
  formData.append('image', blob, 'image.png');

  // Send the API request expecting plain text
  fetch('http://im2latex.ru/convert-to-latex', {
    method: 'POST',
    body: formData,
    headers: {
      'Accept': 'text/plain'
    }
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('API request failed with status ' + response.status);
      }
      return response.text();
    })
    .then(plainLatex => {
      // Save to storage as both latexCode and apiResponse (for UI tab display)
      chrome.storage.local.set({
        latexCode: plainLatex,
        apiResponse: plainLatex
      }, function () {
        sendResponse({
          status: "success",
          latexCode: plainLatex,
          apiResponse: plainLatex
        });
      });
    })
    .catch(error => {
      sendResponse({
        status: "error",
        error: error.toString()
      });
    });
}

// Create extension icon
const iconCanvas = new OffscreenCanvas(16, 16);
const ctx = iconCanvas.getContext('2d');

// Draw a simple LaTeX-like icon
function drawIcon() {
  // Clear canvas
  ctx.clearRect(0, 0, 16, 16);

  // Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 16, 16);

  // Draw "Σ" symbol
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 12px serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Σ', 8, 8);

  return iconCanvas.transferToImageBitmap();
}

// No need to set the icon here as we defined it in the manifest
