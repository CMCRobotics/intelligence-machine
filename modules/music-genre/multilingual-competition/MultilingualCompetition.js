import React, { useState } from 'react';
import { Trophy, Sparkles } from 'lucide-react';

export default function WordCompetition() {
  const [topic, setTopic] = useState('');
  const [numWords, setNumWords] = useState(3);
  const [words, setWords] = useState(['', '', '']);
  const [results, setResults] = useState(null);

  const handleNumWordsChange = (num) => {
    setNumWords(num);
    const newWords = Array(num).fill('').map((_, i) => words[i] || '');
    setWords(newWords);
    setResults(null);
  };

  const handleWordChange = (index, value) => {
    const newWords = [...words];
    newWords[index] = value;
    setWords(newWords);
  };

  const analyzeMultilinguisticIntersection = (word) => {
    // Analyze linguistic elements common across multiple languages
    const commonPrefixes = ['inter', 'multi', 'trans', 'pre', 'post', 'anti', 'auto', 'co', 'de', 're', 'sub', 'super', 'uni', 'bi', 'tri'];
    const commonSuffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 'able', 'ible', 'al', 'ial', 'ic', 'ive', 'ous', 'ful', 'less'];
    const latinRoots = ['act', 'port', 'dict', 'duc', 'fac', 'ject', 'mis', 'mob', 'mot', 'ped', 'pel', 'pend', 'pos', 'rupt', 'scrib', 'sequ', 'spec', 'sta', 'tend', 'ter', 'tract', 'ven', 'vert', 'vid', 'vis', 'voc'];
    
    const lowerWord = word.toLowerCase();
    let score = 3; // Base score
    
    // Check for common prefixes/suffixes (Romance/Germanic overlap)
    commonPrefixes.forEach(prefix => {
      if (lowerWord.startsWith(prefix)) score += 1.5;
    });
    commonSuffixes.forEach(suffix => {
      if (lowerWord.endsWith(suffix)) score += 1.5;
    });
    
    // Check for Latin roots (understood across Romance languages)
    latinRoots.forEach(root => {
      if (lowerWord.includes(root)) score += 1;
    });
    
    // Bonus for simple phonetic structure (universally pronounceable)
    const hasComplexClusters = /[bcdfghjklmnpqrstvwxyz]{3,}/i.test(word);
    if (!hasComplexClusters) score += 1;
    
    // Check for vowel-consonant balance (easier cross-linguistic pronunciation)
    const vowels = (word.match(/[aeiou]/gi) || []).length;
    const consonants = (word.match(/[bcdfghjklmnpqrstvwxyz]/gi) || []).length;
    const ratio = consonants > 0 ? vowels / consonants : 0;
    if (ratio >= 0.4 && ratio <= 1.5) score += 1;
    
    return Math.min(Math.round(score), 10);
  };

  const analyzeAdherence = (word, topic) => {
    // Analyze semantic relevance to topic
    const wordLower = word.toLowerCase();
    const topicLower = topic.toLowerCase();
    const topicWords = topicLower.split(/\s+/);
    
    let score = 2; // Base score
    
    // Direct substring match
    if (wordLower.includes(topicLower) || topicLower.includes(wordLower)) {
      score += 4;
    }
    
    // Check for individual topic word matches
    topicWords.forEach(topicWord => {
      if (topicWord.length > 2 && wordLower.includes(topicWord)) {
        score += 2;
      }
    });
    
    // Check for shared starting letters (suggests thematic connection)
    if (wordLower[0] === topicLower[0]) score += 1;
    
    // Phonetic similarity (rough approximation)
    const sharedChars = new Set([...wordLower]).size;
    const topicChars = new Set([...topicLower]).size;
    const overlap = [...new Set([...wordLower])].filter(c => topicLower.includes(c)).length;
    const similarity = topicChars > 0 ? overlap / topicChars : 0;
    score += similarity * 3;
    
    return Math.min(Math.round(score), 10);
  };

  const analyzeConcision = (word) => {
    const length = word.length;
    let score = 10;
    
    // Optimal length: 5-8 characters
    if (length >= 5 && length <= 8) {
      score = 10;
    } else if (length >= 4 && length <= 10) {
      score = 8;
    } else if (length >= 3 && length <= 12) {
      score = 6;
    } else if (length >= 2 && length <= 15) {
      score = 4;
    } else if (length > 15) {
      score = Math.max(1, 10 - Math.floor((length - 15) / 2));
    } else {
      score = 2;
    }
    
    // Penalty for unnecessary complexity
    const hasNumbers = /\d/.test(word);
    const hasSpecialChars = /[^a-zA-Z0-9]/.test(word);
    if (hasNumbers || hasSpecialChars) score = Math.max(1, score - 2);
    
    return score;
  };

  const evaluateWords = () => {
    if (!topic.trim()) {
      alert('Please enter a topic!');
      return;
    }
    
    const filledWords = words.filter(w => w.trim());
    if (filledWords.length < 2) {
      alert('Please enter at least 2 words to compete!');
      return;
    }

    const evaluations = words.map((word, index) => {
      if (!word.trim()) return null;
      
      const multilingual = analyzeMultilinguisticIntersection(word);
      const adherence = analyzeAdherence(word, topic);
      const concision = analyzeConcision(word);
      const total = multilingual + adherence + concision;
      
      return {
        word,
        index,
        scores: {
          multilingual,
          adherence,
          concision,
          total
        }
      };
    }).filter(Boolean);

    evaluations.sort((a, b) => b.scores.total - a.scores.total);
    setResults(evaluations);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <Sparkles className="w-8 h-8 text-purple-600" />
            <h1 className="text-3xl font-bold text-gray-800">Multilingual Word Competition</h1>
          </div>
          
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Topic / Concept
              </label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Enter the concept to be described..."
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Number of Competing Words
              </label>
              <div className="flex gap-2">
                {[2, 3, 4, 5, 6].map(num => (
                  <button
                    key={num}
                    onClick={() => handleNumWordsChange(num)}
                    className={`px-6 py-2 rounded-lg font-semibold transition-all ${
                      numWords === num
                        ? 'bg-purple-600 text-white shadow-lg scale-105'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Enter Competing Words
              </label>
              <div className="space-y-3">
                {words.map((word, index) => (
                  <input
                    key={index}
                    type="text"
                    value={word}
                    onChange={(e) => handleWordChange(index, e.target.value)}
                    placeholder={`Word ${index + 1}`}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none transition-colors"
                  />
                ))}
              </div>
            </div>

            <button
              onClick={evaluateWords}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 rounded-lg font-bold text-lg hover:from-purple-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl"
            >
              Evaluate Words
            </button>
          </div>
        </div>

        {results && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Trophy className="w-8 h-8 text-yellow-500" />
              <h2 className="text-2xl font-bold text-gray-800">Results</h2>
            </div>

            <div className="space-y-4">
              {results.map((result, index) => (
                <div
                  key={result.index}
                  className={`p-6 rounded-xl border-2 ${
                    index === 0
                      ? 'bg-gradient-to-r from-yellow-50 to-amber-50 border-yellow-400'
                      : 'bg-gray-50 border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      {index === 0 && <Trophy className="w-6 h-6 text-yellow-600" />}
                      <h3 className="text-xl font-bold text-gray-800">
                        {index === 0 ? '🏆 Winner: ' : `#${index + 1}: `}
                        {result.word}
                      </h3>
                    </div>
                    <div className="text-2xl font-bold text-purple-600">
                      {result.scores.total}/30
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-600 font-semibold mb-1">
                        Multilinguistic
                      </div>
                      <div className="text-lg font-bold text-blue-600">
                        {result.scores.multilingual}/10
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-gray-600 font-semibold mb-1">
                        Adherence
                      </div>
                      <div className="text-lg font-bold text-green-600">
                        {result.scores.adherence}/10
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-gray-600 font-semibold mb-1">
                        Concision
                      </div>
                      <div className="text-lg font-bold text-orange-600">
                        {result.scores.concision}/10
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}