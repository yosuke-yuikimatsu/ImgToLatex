
import React, { createContext, useContext, useState } from "react";
import { toast } from "sonner";

// Define the Chrome API types for TypeScript
declare global {
  interface Window {
    chrome?: {
      runtime: {
        sendMessage: (message: any, callback?: (response: any) => void) => void;
      };
      storage: {
        local: {
          set: (items: Object, callback?: () => void) => void;
          get: (keys: string | string[] | Object | null, callback: (items: { [key: string]: any }) => void) => void;
        };
      };
    };
  }
}

interface LatexContextType {
  latexCode: string;
  capturedImage: string | null;
  isLoading: boolean;
  error: string | null;
  apiResponse: any | null;
  setLatexCode: (code: string) => void;
  setCapturedImage: (imageUrl: string) => void;
  fetchLatexFromSelection: (dataUrl: string) => Promise<void>;
  copyToClipboard: () => void;
  setApiResponse: (response: any) => void;
}

const LatexContext = createContext<LatexContextType | undefined>(undefined);

export const useLatex = () => {
  const context = useContext(LatexContext);
  if (!context) {
    throw new Error("useLatex must be used within a LatexProvider");
  }
  return context;
};

export const LatexProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [latexCode, setLatexCode] = useState<string>("\\sqrt{a^2 + b^2}");
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [apiResponse, setApiResponse] = useState<any | null>(null);

  // Send image to the LaTeX conversion API
  const fetchLatexFromSelection = async (dataUrl: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // If we're in a Chrome extension environment, use the chrome API
      if (window.chrome && window.chrome.runtime) {
        // Send message to background script to handle the API call
        window.chrome.runtime.sendMessage({
          action: "sendToApi",
          dataUrl: dataUrl
        }, (response) => {
          if (response.error) {
            setError(response.error);
            console.error("API Error:", response.error);
            toast.error("Failed to convert image");
          } else {
            setLatexCode(response.latexCode);
            setApiResponse(response.apiResponse);
            toast.success("LaTeX code generated successfully!");
          }
          setIsLoading(false);
        });
      } else {
        // For the React demo version, simulate API call
        console.log("Demo mode: Simulating API call with image");
        
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Mock response
        const mockApiResponse = {
          latex_code: "\\[\\frac{y_n}{y_{n-1}} = \\frac{\\left(1 + \\frac{1}{n}\\right)^{n+1}}{\\left(1 + \\frac{1}{n-1}\\right)^{n}} = \\frac{\\left(1 + \\frac{1}{n}\\right)^{n+1}}{\\left(\\frac{n}{n-1}\\right)^{n}}\\]"
        };
        
        setLatexCode(mockApiResponse.latex_code);
        setApiResponse(mockApiResponse);
        setIsLoading(false);
      }
    } catch (err) {
      setError("Failed to process image. Please try again.");
      console.error("Error processing image:", err);
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(latexCode)
      .then(() => {
        toast.success("LaTeX code copied to clipboard!");
      })
      .catch((err) => {
        toast.error("Failed to copy code");
        console.error("Failed to copy:", err);
      });
  };

  return (
    <LatexContext.Provider
      value={{
        latexCode,
        capturedImage,
        isLoading,
        error,
        apiResponse,
        setLatexCode,
        setCapturedImage,
        fetchLatexFromSelection,
        copyToClipboard,
        setApiResponse,
      }}
    >
      {children}
    </LatexContext.Provider>
  );
};
