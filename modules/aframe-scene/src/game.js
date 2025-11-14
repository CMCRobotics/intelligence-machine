import { render } from 'lit-html';
import { html } from 'lit';
import './lit-templates/GameViewManager.js';

// Polyfill global Buffer
import { Buffer } from 'buffer';
window.Buffer = Buffer;

// Define the lit-html template function
const renderGameManager = () => {
    return html`
        <game-view-manager></game-view-manager>
    `;
};

// Find the container and render the scene directly
const container = document.getElementById('game-container');
if (container) {
    render(renderGameManager(), container);
} else {
    console.error("Could not find #game-container to render the scene.");
}
