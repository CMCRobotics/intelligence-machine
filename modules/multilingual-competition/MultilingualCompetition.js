import React, { useState } from 'react';
import { Trophy, Sparkles } from 'lucide-react';

export default function WordCompetition() {
  const [topic, setTopic] = useState('');
  const [numWords, setNumWords] = useState(3);
  const [words, setWords] = useState(['', '', '']);
  const [results, setResults] = useState(null);
  const [aiExplanation, setAiExplanation] = useState('');
  const [loadingExplanation, setLoadingExplanation] = useState(false);

  const handleNumWordsChange = (num) => {
    setNumWords(num);
    const newWords = Array(num).fill('').map((_, i) => words[i] || '');
    setWords(newWords);
    setResults(null);
    setAiExplanation('');
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

  const generateExplanation = (winner, allWords) => {
    const strengths = [];
    const comparisons = [];
    
    // Identify strengths
    if (winner.scores.multilingual >= 8) {
      strengths.push('excellent multilinguistic appeal');
    } else if (winner.scores.multilingual >= 6) {
      strengths.push('good cross-linguistic recognizability');
    }
    
    if (winner.scores.adherence >= 8) {
      strengths.push('strong adherence to the topic');
    } else if (winner.scores.adherence >= 6) {
      strengths.push('clear connection to the concept');
    }
    
    if (winner.scores.concision >= 8) {
      strengths.push('optimal concision');
    } else if (winner.scores.concision >= 6) {
      strengths.push('good length and clarity');
    }
    
    // Compare with runner-ups
    if (allWords.length > 1) {
      const runnerUp = allWords[1];
      const scoreDiff = winner.scores.total - runnerUp.scores.total;
      
      if (scoreDiff >= 5) {
        comparisons.push(`significantly outperforming "${runnerUp.word}"`);
      } else if (scoreDiff >= 2) {
        comparisons.push(`edging out "${runnerUp.word}"`);
      } else {
        comparisons.push(`narrowly beating "${runnerUp.word}"`);
      }
      
      // Mention specific advantage
      const advantages = [];
      if (winner.scores.multilingual > runnerUp.scores.multilingual + 1) {
        advantages.push('better multilinguistic reach');
      }
      if (winner.scores.adherence > runnerUp.scores.adherence + 1) {
        advantages.push('stronger topic alignment');
      }
      if (winner.scores.concision > runnerUp.scores.concision + 1) {
        advantages.push('superior concision');
      }
      
      if (advantages.length > 0) {
        comparisons.push('particularly due to ' + advantages.join(' and '));
      }
    }
    
    let explanation = `"${winner.word}" wins with ${strengths.join(', ')}`;
    if (comparisons.length > 0) {
      explanation += ', ' + comparisons.join(', ');
    }
    explanation += '.';
    
    return explanation;
  };

  const generateAIExplanation = async (winner, allWords, topicText) => {
    setLoadingExplanation(true);
    
    const prompt = `You are judging a multilingual word competition. The topic is "${topicText}".

The competing words and their scores are:
${allWords.map((w, i) => `${i + 1}. "${w.word}" - Total: ${w.scores.total}/30 (Multilinguistic: ${w.scores.multilingual}/10, Adherence: ${w.scores.adherence}/10, Concision: ${w.scores.concision}/10)`).join('\n')}

The winner is "${winner.word}". 

Explain in 2-3 sentences why "${winner.word}" won. Focus on:
- Its multilinguistic appeal (works across Italian, German, Spanish, English, French, etc.)
- How well it captures the topic
- Its conciseness and clarity
- How it compares to the other words

Be specific and insightful. Keep it concise.`;

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': 'YOUR_API_KEY_HERE', // User needs to add their API key
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 200,
          messages: [{
            role: 'user',
            content: prompt
          }]
        })
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();
      const explanation = data.content[0].text;
      setAiExplanation(explanation);
    } catch (error) {
      console.error('Error getting AI explanation:', error);
      setAiExplanation('AI explanation unavailable. Please add your Anthropic API key to enable this feature.');
    } finally {
      setLoadingExplanation(false);
    }
  };

  const evaluateWords = async () => {
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
    
    // Generate explanation
    const explanation = generateExplanation(evaluations[0], evaluations);
    evaluations[0].explanation = explanation;
    
    setResults(evaluations);
    
    // Generate AI explanation
    await generateAIExplanation(evaluations[0], evaluations, topic);
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

            {aiExplanation && (
              <div className="mb-6 p-5 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border-2 border-purple-300">
                <h3 className="text-sm font-bold text-purple-800 mb-2 flex items-center gap-2">
                  <Sparkles className="w-4 h-4" />
                  AI Analysis
                </h3>
                <p className="text-gray-700 leading-relaxed">{aiExplanation}</p>
              </div>
            )}

            {loadingExplanation && !aiExplanation && (
              <div className="mb-6 p-5 bg-gray-50 rounded-xl border-2 border-gray-300">
                <p className="text-gray-500 italic">Generating AI explanation...</p>
              </div>
            )}

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