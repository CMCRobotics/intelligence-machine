import Reveal from 'reveal.js';
import Markdown from 'reveal.js/plugin/markdown/markdown.esm.js';

// We import the CSS in JS, but Bun will bundle it into a separate CSS file.
// Ensure the HTML has <link rel="stylesheet" href="presentation.css">
import 'reveal.js/dist/reveal.css';
import 'reveal.js/dist/theme/white.css';

const deck = new Reveal({
  plugins: [Markdown],
  // height: 1080,
  // width: 1920,
  progress: false,
  controls: true,
  autoSlideMethod: "right",
  transition: "none",
});

deck.initialize().then(() => {
    console.log("Reveal.js initialized successfully");
}).catch((err) => {
    console.error("Reveal.js initialization failed", err);
});
