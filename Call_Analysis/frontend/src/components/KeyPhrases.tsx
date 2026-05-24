'use client';

import { ThumbsUp, ThumbsDown } from 'lucide-react';

interface KeyPhrasesProps {
  phrases: {
    positive: string[];
    negative: string[];
  };
}

const KeyPhrases = ({ phrases }: KeyPhrasesProps) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Positive Phrases */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <ThumbsUp className="h-5 w-5 text-green-600" />
          <h4 className="text-lg font-semibold text-green-800">Positive Phrases</h4>
        </div>
        <div className="space-y-2">
          {phrases.positive.length > 0 ? (
            phrases.positive.map((phrase, index) => (
              <div
                key={index}
                className="p-3 bg-green-50 border border-green-200 rounded-lg"
              >
                <p className="text-green-800 font-medium">"{phrase}"</p>
              </div>
            ))
          ) : (
            <p className="text-gray-500 italic">No positive phrases detected</p>
          )}
        </div>
      </div>

      {/* Negative Phrases */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <ThumbsDown className="h-5 w-5 text-red-600" />
          <h4 className="text-lg font-semibold text-red-800">Negative Phrases</h4>
        </div>
        <div className="space-y-2">
          {phrases.negative.length > 0 ? (
            phrases.negative.map((phrase, index) => (
              <div
                key={index}
                className="p-3 bg-red-50 border border-red-200 rounded-lg"
              >
                <p className="text-red-800 font-medium">"{phrase}"</p>
              </div>
            ))
          ) : (
            <p className="text-gray-500 italic">No negative phrases detected</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default KeyPhrases;
