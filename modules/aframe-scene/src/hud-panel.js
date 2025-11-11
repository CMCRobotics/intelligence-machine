// src/hud-panel.js

// Create a container for the HUD
const hudContainer = document.createElement('div');
hudContainer.id = 'hud-container';
hudContainer.style.position = 'fixed';
hudContainer.style.top = '0';
hudContainer.style.left = '0';
hudContainer.style.width = '100%';
hudContainer.style.height = '100%';
hudContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'; // Semi-transparent black background
hudContainer.style.pointerEvents = 'auto'; // Allow pointer events to interact with the HUD
hudContainer.style.zIndex = '1000'; // Ensure it's on top of other content
hudContainer.style.display = 'none'; // Initially hidden

// Add some basic content to the HUD
const hudContent = document.createElement('div');
hudContent.style.position = 'absolute';
hudContent.style.top = '50%';
hudContent.style.left = '50%';
hudContent.style.transform = 'translate(-50%, -50%)';
hudContent.style.color = 'white';
hudContent.style.fontSize = '24px';
hudContent.style.textAlign = 'center';
hudContent.textContent = 'HUD Panel Active';
hudContainer.appendChild(hudContent);

// Append the HUD to the document body
document.body.appendChild(hudContainer);

// Function to show the HUD
function showHudPanel() {
  hudContainer.style.display = 'block';
  // Suppress interactions with the A-Frame scene
  // This can be done by setting pointerEvents to 'none' on the A-Frame canvas or its parent
  // For simplicity, we'll assume the HUD container itself handles pointer events.
  // If the A-Frame scene is still interactive, more specific DOM manipulation might be needed.
  // For example, finding the canvas and setting its pointerEvents to 'none'.
  const aframeCanvas = document.querySelector('a-scene') ? document.querySelector('a-scene').canvas : null;
  if (aframeCanvas) {
    aframeCanvas.style.pointerEvents = 'none';
  }
}

// Function to hide the HUD
function hideHudPanel() {
  hudContainer.style.display = 'none';
  // Restore interactions with the A-Frame scene
  const aframeCanvas = document.querySelector('a-scene') ? document.querySelector('a-scene').canvas : null;
  if (aframeCanvas) {
    aframeCanvas.style.pointerEvents = 'auto';
  }
}

// Export functions to control the HUD
export { showHudPanel, hideHudPanel };
