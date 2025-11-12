import Reveal from 'reveal.js';
import Markdown from 'reveal.js/plugin/markdown/markdown.esm.js';
import 'reveal.js/dist/reveal.css';
import 'reveal.js/dist/theme/white.css';


Reveal.initialize({
    plugins: [ Markdown ],
    progress: false,
    controls: true,
    autoSlideMethod: "right",
    transition: "none"
  });
