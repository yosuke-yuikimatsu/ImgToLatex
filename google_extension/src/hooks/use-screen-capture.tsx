
import { useState, useEffect } from "react";
import { useLatex } from "@/contexts/LatexContext";
import { toast } from "sonner";

export const useScreenCapture = () => {
  const [isCapturing, setIsCapturing] = useState(false);
  const { fetchLatexFromSelection, setCapturedImage } = useLatex();

  // Create and manage the selection overlay elements
  useEffect(() => {
    if (!isCapturing) return;

    let startX = 0;
    let startY = 0;
    let endX = 0;
    let endY = 0;
    let isDragging = false;

    // Create overlay elements
    const overlay = document.createElement("div");
    overlay.className = "backdrop";
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 99998;
      background: rgba(0, 0, 0, 0.3);
      backdrop-filter: blur(2px);
      cursor: crosshair;
    `;
    document.body.appendChild(overlay);

    const selectionBox = document.createElement("div");
    selectionBox.className = "selection-area";
    selectionBox.style.cssText = `
      position: fixed;
      z-index: 99999;
      pointer-events: none;
      background: rgba(76, 161, 255, 0.2);
      border: 1px solid rgba(76, 161, 255, 0.8);
      box-shadow: 0 0 10px rgba(76, 161, 255, 0.3);
      display: none;
    `;
    document.body.appendChild(selectionBox);

    // Handle mouse events for selection
    const handleMouseDown = (e) => {
      startX = e.clientX;
      startY = e.clientY;
      endX = e.clientX;
      endY = e.clientY;
      isDragging = true;
      selectionBox.style.display = "block";
      updateSelectionBox();
    };

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      endX = e.clientX;
      endY = e.clientY;
      updateSelectionBox();
    };

    const handleMouseUp = async () => {
      if (!isDragging) return;
      isDragging = false;

      const width = Math.abs(endX - startX);
      const height = Math.abs(endY - startY);

      // Only process if the selection has a reasonable size
      if (width > 10 && height > 10) {
        // Capture the selected area
        const canvas = document.createElement("canvas");
        const left = Math.min(startX, endX);
        const top = Math.min(startY, endY);
        
        canvas.width = width;
        canvas.height = height;
        
        // In a real extension, we would use chrome.tabs.captureVisibleTab
        // For this simulation, we'll just create a canvas with the selection dimensions
        const ctx = canvas.getContext("2d");
        if (ctx) {
          // Create a gradient background to simulate a captured area
          const gradient = ctx.createLinearGradient(0, 0, width, height);
          gradient.addColorStop(0, "#f0f4ff");
          gradient.addColorStop(1, "#e0e7ff");
          ctx.fillStyle = gradient;
          ctx.fillRect(0, 0, width, height);
          
          // Draw a border
          ctx.strokeStyle = "#6366f1";
          ctx.lineWidth = 2;
          ctx.strokeRect(2, 2, width - 4, height - 4);
          
          // Add some text
          ctx.font = "14px Arial";
          ctx.fillStyle = "#4f46e5";
          ctx.textAlign = "center";
          ctx.fillText("Captured Area", width / 2, height / 2);
          ctx.font = "11px Arial";
          ctx.fillStyle = "#6366f1";
          ctx.fillText(`${width} × ${height}px`, width / 2, height / 2 + 20);
        }
        
        const dataUrl = canvas.toDataURL("image/png");
        
        // Clean up
        cleanupSelection();
        
        // Save the captured image
        setCapturedImage(dataUrl);
        
        // Process the captured area (mock LaTeX recognition for demo)
        await fetchLatexFromSelection(dataUrl);
        
        toast.success("Area captured successfully!");
      } else {
        // Selection too small
        toast.warning("Selection area too small. Please try again.");
        cleanupSelection();
      }
    };

    // Update the selection box position and size
    const updateSelectionBox = () => {
      const left = Math.min(startX, endX);
      const top = Math.min(startY, endY);
      const width = Math.abs(endX - startX);
      const height = Math.abs(endY - startY);

      selectionBox.style.left = `${left}px`;
      selectionBox.style.top = `${top}px`;
      selectionBox.style.width = `${width}px`;
      selectionBox.style.height = `${height}px`;
      selectionBox.style.display = "block";
    };

    const cleanupSelection = () => {
      document.body.removeChild(overlay);
      document.body.removeChild(selectionBox);
      setIsCapturing(false);
    };

    // Add event listeners
    overlay.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    // Escape key to cancel
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        cleanupSelection();
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    // Cleanup function
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.removeEventListener("keydown", handleKeyDown);
      
      if (document.body.contains(overlay)) {
        document.body.removeChild(overlay);
      }
      
      if (document.body.contains(selectionBox)) {
        document.body.removeChild(selectionBox);
      }
    };
  }, [isCapturing, fetchLatexFromSelection, setCapturedImage]);

  const startCapture = () => {
    setIsCapturing(true);
  };

  return { startCapture, isCapturing };
};
