import React, { useState } from 'react';
import axios from 'axios';
import './Summarizer.css';

const Summarizer = () => {
    const [inputText, setInputText] = useState('');
    const [summary, setSummary] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSummarize = async () => {
        setLoading(true);
        setError(null);
        setSummary('');

        try {
            const response = await axios.post(
                'https://api-inference.huggingface.co/models/facebook/bart-large-cnn',
                { inputs: inputText },
                {
                    headers: {
                        'Authorization': `Bearer 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`,
                        'Content-Type': 'application/json',
                    },
                }
            );

            setSummary(response.data[0].summary_text); // Get the summary from the response
        } catch (error) {
            setError('Failed to summarize the text.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="summarizer">
            <h2>Text Summarizer</h2>
            <textarea
                rows="10"
                cols="50"
                placeholder="Enter text to summarize..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
            />
            <br />
            <button onClick={handleSummarize} disabled={loading}>
                {loading ? 'Summarizing...' : 'Summarize'}
            </button>
            {error && <div className="error">{error}</div>}
            {summary && (
                <div className="summary">
                    <h3>Summary:</h3>
                    <p>{summary}</p>
                </div>
            )}
        </div>
    );
};

export default Summarizer;
