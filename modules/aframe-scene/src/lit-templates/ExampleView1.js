import { LitElement, html, css } from 'lit';

class ExampleView1 extends LitElement {
  render() {
    return html`
      <div class="example-view">
        <h2>This is Example View 1</h2>
        <p>Content for the first view goes here.</p>
      </div>
    `;
  }

  static styles = css`
    .example-view {
      padding: 20px;
      border: 1px dashed blue;
      background-color: #e0f7fa;
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

customElements.define('example-view-1', ExampleView1);

export { ExampleView1 };
