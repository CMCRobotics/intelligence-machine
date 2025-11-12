const path = require('path');
const webpack = require('webpack');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
  entry: {
    vr: './src/vr.js',
    presentation: './src/presentation.js',
    desktop: './src/desktop.js'
  },
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist')
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      },
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      }
    ]
  },
  plugins: [
    
    new webpack.ProvidePlugin({
      process: 'process/browser',
      Buffer: ['buffer', 'Buffer'],
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: 'public', to: '' }
      ],
    }),
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    }),
    new webpack.NormalModuleReplacementPlugin(/node:/, (resource) => {
      const mod = resource.request.replace(/^node:/, "");
      switch (mod) {
        case "path":
          resource.request = "path-browserify";
          break;
        case "url":
          resource.request = "url/";
          break;
        default:
          throw new Error(`Not found ${mod}`);
      }
    }),
  ],
  resolve: {
    modules: [path.resolve(__dirname, 'src'), 'node_modules'],
    extensions: ['.ts', '.js', '.json', '.tsx'],
    fallback: {
      "fs" : false,
      "ws": false,
      "url": require.resolve("url/"),
      "net": false,
      "tls": false,
      "path": require.resolve("path-browserify"),
      "zlib": require.resolve("browserify-zlib"),
      "stream": require.resolve("stream-browserify"),
      "http": require.resolve("stream-http"),
      "https": require.resolve("https-browserify"),
      "crypto": require.resolve("crypto-browserify"),
      "buffer": require.resolve("buffer/")
    },
    alias: {
      '@cmcrobotics/homie-lit': '@cmcrobotics/homie-lit',
      // Add aliases for pronolab modules to point to their .ts files
      './pronolab/core/session': './pronolab/core/session.ts',
      './pronolab/view/view-manager': './pronolab/view/view-manager.ts',
      './pronolab/view/image-view': './pronolab/view/image-view.ts',
      './pronolab/view/audio-view': './pronolab/view/audio-view.ts',
      './pronolab/view/pose-view': './pronolab/view/pose-view.ts',
    },
    mainFields: ['module', 'browser', 'main']
  },
  devServer: {
    static: {
      directory: path.join(__dirname, 'public'),
    },
    host: '0.0.0.0',
    hot: true,
    compress: true,
    port: 9000,
  }
};
