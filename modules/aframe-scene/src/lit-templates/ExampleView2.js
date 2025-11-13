import { LitElement, html, css } from 'lit';

class ExampleView2 extends LitElement {
  render() {
    return html`
      <div class="example-view">
        <h2>This is Example View 2</h2>
        <p>Content for the second view goes here.</p>
      </div>
    `;
  }

  static styles = css`
    .example-view {
      padding: 20px;
      border: 1px dashed green;
      background-color: #e8f5e9;
      text-align: center;
      width: 80%;
      height: 80%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
  `;
}

customElements.define('example-view-2', ExampleView2);

export { ExampleView2 };
