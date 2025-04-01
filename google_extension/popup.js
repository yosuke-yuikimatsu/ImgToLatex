document.addEventListener('DOMContentLoaded', function () {
  // Tab switching functionality
  const codeTab = document.getElementById('code-tab');
  const previewTab = document.getElementById('preview-tab');
  const apiTab = document.getElementById('api-tab');
  const codeContent = document.getElementById('code-content');
  const previewContent = document.getElementById('preview-content');
  const apiContent = document.getElementById('api-content');
  const tabIndicator = document.querySelector('.tab-indicator');

  // Copy button functionality
  const copyButton = document.getElementById('copy-button');
  const latexCode = document.getElementById('latex-code');

  // Capture button functionality
  const captureButton = document.getElementById('capture-button');

  // LaTeX preview rendering
  const latexPreview = document.getElementById('latex-preview');

  // API response element
  const apiResponseElement = document.getElementById('api-response');

  // Image preview element
  const capturedImagePreview = document.getElementById('captured-image-preview');

  // Calculate tab widths precisely
  const tabWidth = 100 / (apiTab ? 3 : 2);

  // Initialize with tab 1 active
  updateTabIndicator(0);

  // Check if there's stored LaTeX code, API response, and captured image
  chrome.storage.local.get(['latexCode', 'capturedImage', 'apiResponse'], function (result) {
    if (result.latexCode) {
      latexCode.textContent = result.latexCode;
    }

    if (result.capturedImage) {
      capturedImagePreview.src = result.capturedImage;
      capturedImagePreview.style.display = 'block';
    }

    if (result.apiResponse) {
      apiResponseElement.textContent = typeof result.apiResponse === "string" ? result.apiResponse : JSON.stringify(result.apiResponse, null, 2);
    }

    waitForKaTeX().then(() => {
      renderLatexPreview();
    }).catch(error => {
      console.error('Error waiting for KaTeX:', error);
      latexPreview.innerHTML = `<div class="error">Error loading KaTeX library. Please refresh the popup.</div>`;
    });
  });

  // Function to wait until KaTeX is loaded
  function waitForKaTeX() {
    return new Promise((resolve, reject) => {
      const maxAttempts = 10;
      let attempts = 0;

      function checkKaTeX() {
        attempts++;
        if (typeof window.katex !== 'undefined') {
          resolve();
        } else if (attempts >= maxAttempts) {
          console.error('KaTeX failed to load after multiple attempts');
          reject(new Error('KaTeX failed to load'));
        } else {
          setTimeout(checkKaTeX, 300);
        }
      }

      checkKaTeX();
    });
  }

  // Tab switching
  codeTab.addEventListener('click', function () {
    activateTab(codeTab, codeContent);
    deactivateTab(previewTab, previewContent);
    if (apiTab) deactivateTab(apiTab, apiContent);
    updateTabIndicator(0);
  });

  previewTab.addEventListener('click', function () {
    activateTab(previewTab, previewContent);
    deactivateTab(codeTab, codeContent);
    if (apiTab) deactivateTab(apiTab, apiContent);
    updateTabIndicator(1);

    // Ensure KaTeX is loaded before rendering
    waitForKaTeX().then(() => {
      renderLatexPreview();
    }).catch(error => {
      console.error('Error waiting for KaTeX:', error);
      latexPreview.innerHTML = `<div class="error">Error loading KaTeX library. Please refresh the popup.</div>`;
    });
  });

  if (apiTab) {
    apiTab.addEventListener('click', function () {
      activateTab(apiTab, apiContent);
      deactivateTab(codeTab, codeContent);
      deactivateTab(previewTab, previewContent);
      updateTabIndicator(2);
    });
  }

  // Copy to clipboard
  copyButton.addEventListener('click', function () {
    navigator.clipboard.writeText(latexCode.textContent).then(function () {
      const originalText = copyButton.querySelector('.button-text').textContent;
      copyButton.querySelector('.button-text').textContent = 'Copied!';

      setTimeout(function () {
        copyButton.querySelector('.button-text').textContent = originalText;
      }, 2000);
    });
  });

  // Capture screen area
  captureButton.addEventListener('click', function () {
    console.log("Capture button clicked");

    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      if (!tabs || tabs.length === 0) {
        console.error("No active tab found");
        return;
      }

      const activeTab = tabs[0];

      // Check if content script is already loaded
      chrome.tabs.sendMessage(activeTab.id, { action: "ping" }, function (response) {
        // If the content script doesn't respond, inject it first
        if (chrome.runtime.lastError) {
          console.log("Content script not loaded, injecting it now");

          chrome.scripting.executeScript({
            target: { tabId: activeTab.id },
            files: ['content.js']
          }, function () {
            if (chrome.runtime.lastError) {
              console.error("Script injection error:", chrome.runtime.lastError);
              return;
            }

            // Now send the message after script is injected
            startSelectionProcess(activeTab.id);
          });
        } else {
          // Content script is already loaded, proceed directly
          startSelectionProcess(activeTab.id);
        }
      });
    });
  });

  // Function to start the selection process
  function startSelectionProcess(tabId) {
    // Send message to content script to start area selection
    chrome.tabs.sendMessage(tabId, { action: "startSelection" }, function (response) {
      if (chrome.runtime.lastError) {
        console.error("Error sending message:", chrome.runtime.lastError);
      } else {
        console.log("Selection response:", response);
        // Close the popup to show the webpage for selection
        window.close();
      }
    });
  }

  // Helper functions
  function activateTab(tab, content) {
    tab.classList.add('active');
    content.classList.add('active');
  }

  function deactivateTab(tab, content) {
    tab.classList.remove('active');
    content.classList.remove('active');
  }

  function updateTabIndicator(tabIndex) {
    // Fix the tab indicator width and position calculation
    const width = tabWidth;
    const position = tabIndex * width * 2.87;

    tabIndicator.style.width = `${width}%`;
    tabIndicator.style.transform = `translateX(${position}%)`;
  }

  function renderLatexPreview() {
    const code = latexCode.textContent;
    try {
      latexPreview.innerHTML = '';
      window.katex.render(code, latexPreview, {
        displayMode: true,
        throwOnError: false
      });
    } catch (e) {
      latexPreview.innerHTML = `<div class="error">Error rendering LaTeX: ${e.message}</div>`;
    }
  }
});
