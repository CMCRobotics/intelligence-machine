import Reveal from 'reveal.js';
import Markdown from 'reveal.js/plugin/markdown/markdown.esm.js';
import 'reveal.js/dist/reveal.css';
import 'reveal.js/dist/theme/white.css';


Reveal.initialize({
    plugins: [ Markdown ],
    width: 1920,
    height: 1080,
    margin: 0.1,
    progress: false,
    controls: true,
    autoSlideMethod: "right",
    transition: "none",
    autoSlide: 4000,
    // loop: true
  }).then(() => {
    // $(document).ready(
    //     function(){
    //         // $('body').on('submit','form',function() {
    //         //     var sessionCode = $('#session_code')[0].value;
    //         //     $("section[data-microsquad]").each(
    //         //         function(){
    //         //             var iframeUrl = new URL( $(this).data("background-iframe") );
    //         //             iframeUrl.searchParams.set("sc", sessionCode);
    //         //             $(this).data("background-iframe",iframeUrl.href);
    //         //             console.log("Now "+$(this).data("background-iframe"));
    //         //         }
    //         //     );
    //         //     return false;
    //         // });
    //     } 
    // );
});
