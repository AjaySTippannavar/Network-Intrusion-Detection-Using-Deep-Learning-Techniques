
// NIDS Dashboard v6 – Stable Charts + ROC & Loss Curves + Accuracy Summary
console.log("Dashboard JS loaded ✅");

function resetDiv(divId) {
  const div = document.getElementById(divId);
  if (!div) return;
  div.style.height = "420px";
  div.style.width = "100%";
  div.style.overflow = "hidden";
  div.innerHTML = "";
  Plotly.purge(divId);
}

function renderPlot(divId, data, layout) {
  resetDiv(divId);
  Plotly.react(divId, data, layout, {
    staticPlot: false,
    responsive: true,
    displayModeBar: false
  });
}

function showAccuracy(acc) {
  let box = document.getElementById("accuracy-box");
  if (!box) {
    box = document.createElement("div");
    box.id = "accuracy-box";
    box.style.textAlign = "center";
    box.style.fontSize = "18px";
    box.style.fontWeight = "600";
    box.style.margin = "8px 0";
    box.style.color = "#7fff7f";
    document.getElementById("bar").parentElement.prepend(box);
  }
  box.innerHTML = `Overall Accuracy: <span style="color:#00ffaa">${(acc * 100).toFixed(2)}%</span>`;
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("btn-eval");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const dataset = document.getElementById("eval-dataset").value;
    const model = document.getElementById("eval-model").value;
    try {
      const r = await fetch(`/api/report/${dataset}/${model}`);
      const data = await r.json();

      const labels = Object.keys(data.metrics);
      const prec = labels.map(k => data.metrics[k].precision);
      const rec = labels.map(k => data.metrics[k].recall);
      const f1 = labels.map(k => data.metrics[k].f1);
      const acc = data.overall_accuracy || data.accuracy || 0;
      showAccuracy(acc);

      const traces = [
        { x: labels, y: prec, type: "bar", name: "Precision" },
        { x: labels, y: rec, type: "bar", name: "Recall" },
        { x: labels, y: f1, type: "bar", name: "F1 Score" },
        {
          x: labels,
          y: new Array(labels.length).fill(acc),
          type: "scatter",
          mode: "lines",
          name: "Accuracy",
          line: { color: "#00ffaa", dash: "dot", width: 2 }
        }
      ];

      const layoutBar = {
        title: `${dataset} / ${model} Metrics`,
        margin: { t: 40, l: 40, r: 10, b: 50 },
        height: 400,
        autosize: false,
        fixedrange: true,
        showlegend: true,
        legend: { orientation: "h", x: 0.25, y: -0.3 },
        yaxis: { title: "Score", range: [0, 1] }
      };

      renderPlot("bar", traces, layoutBar);

      const conf = data.confusion;
      const confData = [{
        z: conf.matrix,
        x: conf.labels,
        y: conf.labels,
        type: "heatmap",
        colorscale: "Viridis"
      }];
      const confLayout = {
        title: "Confusion Matrix",
        height: 400,
        autosize: false,
        fixedrange: true,
        margin: { t: 40, l: 60, r: 20, b: 50 }
      };
      renderPlot("confusion", confData, confLayout);

      // ROC Curve
      if (data.roc) {
        const rocData = [{
          x: data.roc.fpr,
          y: data.roc.tpr,
          type: "scatter",
          mode: "lines",
          name: `AUC = ${data.roc.auc.toFixed(3)}`
        }];
        const rocLayout = {
          title: "ROC Curve",
          height: 400,
          autosize: false,
          xaxis: { title: "False Positive Rate" },
          yaxis: { title: "True Positive Rate", range: [0, 1] },
          showlegend: true
        };
        renderPlot("roc", rocData, rocLayout);
      }

      // Training vs Validation Loss
      if (data.loss_curve) {
        const lossData = [
          { x: data.loss_curve.epochs, y: data.loss_curve.train_loss, type: "scatter", mode: "lines+markers", name: "Train Loss" },
          { x: data.loss_curve.epochs, y: data.loss_curve.val_loss, type: "scatter", mode: "lines+markers", name: "Val Loss" }
        ];
        const lossLayout = {
          title: "Training vs Validation Loss",
          height: 400,
          autosize: false,
          xaxis: { title: "Epochs" },
          yaxis: { title: "Loss" },
          showlegend: true
        };
        renderPlot("loss", lossData, lossLayout);
      }
    } catch (err) {
      console.error("Report load error", err);
      alert("Error loading report.");
    }
  });
});
