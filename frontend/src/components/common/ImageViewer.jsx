import React from 'react';
import { X } from 'lucide-react';

export default function ImageViewer({ src, alt = 'Image', onClose }) {
  if (!src) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70" onClick={onClose}>
      <div className="relative max-w-3xl w-full mx-4" onClick={(e) => e.stopPropagation()}>
        <button onClick={onClose} className="absolute right-2 top-2 bg-white rounded-full p-1 shadow">
          <X size={20} />
        </button>
        <div className="bg-white rounded overflow-hidden">
          <img src={src} alt={alt} className="w-full h-[70vh] object-contain bg-black" />
        </div>
      </div>
    </div>
  );
}
