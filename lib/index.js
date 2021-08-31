module.exports = [
  {
    id: 'k3d',
    autoStart: true,
    activate: function (app) {
      console.log(
        'JupyterLab extension k3d is activated!'
      );
      console.log(app.commands);
    }
  }
];
