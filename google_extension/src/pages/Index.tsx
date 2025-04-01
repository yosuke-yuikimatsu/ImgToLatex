
import { useState } from "react";
import { motion } from "framer-motion";
import { useToast } from "@/components/ui/use-toast";
import { useLatex } from "@/contexts/LatexContext";
import CaptureButton from "@/components/CaptureButton";

const Index = () => {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<"code" | "preview">("code");
  const { latexCode, copyToClipboard, capturedImage } = useLatex();

  const handleCopy = () => {
    copyToClipboard();
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-gray-50 to-gray-100 p-4">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-md mx-auto"
      >
        <div className="glass-morphism rounded-xl overflow-hidden shadow-xl">
          <div className="p-6">
            <h1 className="text-2xl font-medium text-center text-gray-900 mb-6">LaTeX Snapshot</h1>
            
            {/* Tab Navigation */}
            <div className="bg-gray-100 rounded-lg p-1 mb-6 relative">
              <div className="grid grid-cols-2 relative z-10">
                <button 
                  onClick={() => setActiveTab("code")} 
                  className={`py-2 text-sm font-medium text-center transition-colors duration-200 ${activeTab === "code" ? "text-gray-900" : "text-gray-500"}`}
                >
                  Code
                </button>
                <button 
                  onClick={() => setActiveTab("preview")} 
                  className={`py-2 text-sm font-medium text-center transition-colors duration-200 ${activeTab === "preview" ? "text-gray-900" : "text-gray-500"}`}
                >
                  Preview
                </button>
              </div>
              
              {/* Sliding indicator */}
              <motion.div 
                className="absolute top-1 left-1 bottom-1 rounded-md bg-white shadow-sm tab-indicator"
                initial={false}
                animate={{ 
                  x: activeTab === "code" ? 0 : "100%" 
                }}
                style={{ width: "calc(50% - 2px)" }}
              />
            </div>
            
            {/* Content Area */}
            <div className="relative">
              {/* Code Tab */}
              <motion.div 
                initial={false}
                animate={{ 
                  opacity: activeTab === "code" ? 1 : 0,
                  x: activeTab === "code" ? 0 : -20,
                  pointerEvents: activeTab === "code" ? "auto" : "none"
                }}
                transition={{ duration: 0.3 }}
                className="mb-4"
              >
                <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm overflow-x-auto mb-4">
                  {latexCode}
                </div>
                
                <button 
                  onClick={handleCopy}
                  className="w-full bg-gray-900 text-white py-3 rounded-lg font-medium transition-all hover:bg-gray-800 active:scale-[0.98]"
                >
                  Copy to Clipboard
                </button>
              </motion.div>
              
              {/* Preview Tab */}
              <motion.div 
                initial={false}
                animate={{ 
                  opacity: activeTab === "preview" ? 1 : 0,
                  x: activeTab === "preview" ? 0 : 20,
                  pointerEvents: activeTab === "preview" ? "auto" : "none" 
                }}
                transition={{ duration: 0.3 }}
                className="absolute top-0 left-0 w-full"
                style={{ display: activeTab === "preview" ? "block" : "none" }}
              >
                <div className="bg-gray-50 rounded-lg p-8 flex items-center justify-center min-h-[150px] flex-col">
                  {capturedImage && (
                    <img 
                      src={capturedImage} 
                      alt="Captured Area" 
                      className="max-w-full max-h-[120px] mb-4 border border-gray-200 rounded"
                    />
                  )}
                  <img 
                    src={`https://latex.codecogs.com/svg.latex?${encodeURIComponent(latexCode)}`}
                    alt="LaTeX Formula" 
                    className="max-w-full max-h-[120px]"
                  />
                </div>
              </motion.div>
            </div>
          </div>
          
          <div className="px-6 py-4 border-t border-gray-100">
            <CaptureButton />
          </div>
        </div>
        
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            This is a demo of the LaTeX Snapshot Chrome Extension
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Currently using a mock API with the sample formula
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default Index;
