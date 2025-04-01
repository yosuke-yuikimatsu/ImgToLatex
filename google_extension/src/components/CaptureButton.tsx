
import React from "react";
import { Button } from "@/components/ui/button";
import { Crop } from "lucide-react";
import { useScreenCapture } from "@/hooks/use-screen-capture";

const CaptureButton = () => {
  const { startCapture, isCapturing } = useScreenCapture();

  return (
    <Button 
      onClick={startCapture} 
      disabled={isCapturing}
      className="w-full transition-all duration-300 bg-gradient-to-r from-blue-500 to-indigo-500 text-white py-3 rounded-lg font-medium shadow-lg shadow-blue-100 hover:shadow-xl hover:shadow-blue-200 hover:-translate-y-[1px] active:translate-y-[0px] gap-2"
    >
      <Crop className="w-4 h-4" />
      <span>{isCapturing ? "Selecting area..." : "Select Screen Area"}</span>
    </Button>
  );
};

export default CaptureButton;
