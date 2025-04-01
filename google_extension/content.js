
// This script will be automatically injected into every page
// It will listen for messages from the background script

// Listen for message from background script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "startSelection") {
    startAreaSelection();
    sendResponse({status: "Selection started"});
  } else if (request.action === "ping") {
    // Simple ping to check if content script is loaded
    sendResponse({status: "Content script is loaded"});
  }
  return true;
});

// Function to start area selection
function startAreaSelection() {
  if (document.getElementById('latex-selection-backdrop')) {
    return; // Selection already in progress
  }
  
  // Create backdrop
  const backdrop = document.createElement('div');
  backdrop.id = 'latex-selection-backdrop';
  backdrop.className = 'backdrop';
  document.body.appendChild(backdrop);
  
  // Create selection area
  const selection = document.createElement('div');
  selection.className = 'selection-area';
  document.body.appendChild(selection);
  
  // Selection variables
  let startX, startY, isSelecting = false;
  
  // Mouse down - start selection
  backdrop.addEventListener('mousedown', function(e) {
    isSelecting = true;
    startX = e.clientX;
    startY = e.clientY;
    
    selection.style.left = startX + 'px';
    selection.style.top = startY + 'px';
    selection.style.width = '0px';
    selection.style.height = '0px';
    selection.style.display = 'block';
  });
  
  // Mouse move - update selection size
  backdrop.addEventListener('mousemove', function(e) {
    if (!isSelecting) return;
    
    const currentX = e.clientX;
    const currentY = e.clientY;
    
    const width = currentX - startX;
    const height = currentY - startY;
    
    // Handle negative dimensions (selecting up/left)
    if (width < 0) {
      selection.style.left = currentX + 'px';
      selection.style.width = Math.abs(width) + 'px';
    } else {
      selection.style.left = startX + 'px';
      selection.style.width = width + 'px';
    }
    
    if (height < 0) {
      selection.style.top = currentY + 'px';
      selection.style.height = Math.abs(height) + 'px';
    } else {
      selection.style.top = startY + 'px';
      selection.style.height = height + 'px';
    }
  });
  
  // Mouse up - end selection
  backdrop.addEventListener('mouseup', function() {
    if (!isSelecting) return;
    isSelecting = false;
    
    // Get selection coordinates
    const rect = selection.getBoundingClientRect();
    const captureData = {
      x: rect.left,
      y: rect.top,
      width: rect.width,
      height: rect.height,
      devicePixelRatio: window.devicePixelRatio
    };
    
    // Remove selection UI
    selection.remove();
    backdrop.remove();
    
    // Send message to background script with capture data
    chrome.runtime.sendMessage({
      action: "captureComplete", 
      data: captureData
    }, function(response) {
      console.log("Capture response:", response);
    });
  });
  
  // User can press Escape to cancel selection
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && (backdrop.parentNode || selection.parentNode)) {
      selection.remove();
      backdrop.remove();
    }
  }, {once: true});
}
