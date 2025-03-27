
// Auto-converted from TSX using Babel - Browser Compatible Version
(function() {
    // Get React from global scope
    var React = window.React;

    // Make React hooks available
    var useState = React.useState;
    var useEffect = React.useEffect;
    var useRef = React.useRef;

    // Simple JSX runtime replacement
    var jsx = function(type, props, key, children) {
        var newProps = props || {};
        if (children !== undefined) {
            newProps.children = children;
        }
        return React.createElement(type, newProps, children);
    };

    var jsxs = function(type, props, key, children) {
        return jsx(type, props, key, children);
    };

    // Helper functions from Babel
    function _extends() {
        _extends = Object.assign || function(target) {
            for (var i = 1; i < arguments.length; i++) {
                var source = arguments[i];
                for (var key in source) {
                    if (Object.prototype.hasOwnProperty.call(source, key)) {
                        target[key] = source[key];
                    }
                }
            }
            return target;
        };
        return _extends.apply(this, arguments);
    }

    // Babel transpiled component with export statements removed
function _slicedToArray(r, e) { return _arrayWithHoles(r) || _iterableToArrayLimit(r, e) || _unsupportedIterableToArray(r, e) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(r, a) { if (r) { if ("string" == typeof r) return _arrayLikeToArray(r, a); var t = {}.toString.call(r).slice(8, -1); return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0; } }
function _arrayLikeToArray(r, a) { (null == a || a > r.length) && (a = r.length); for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e]; return n; }
function _iterableToArrayLimit(r, l) { var t = null == r ? null : "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (null != t) { var e, n, i, u, a = [], f = !0, o = !1; try { if (i = (t = t.call(r)).next, 0 === l) { if (Object(t) !== t) return; f = !1; } else for (; !(f = (e = i.call(t)).done) && (a.push(e.value), a.length !== l); f = !0); } catch (r) { o = !0, n = r; } finally { try { if (!f && null != t.return && (u = t.return(), Object(u) !== u)) return; } finally { if (o) throw n; } } return a; } }
function _arrayWithHoles(r) { if (Array.isArray(r)) return r; }
// Using global React object (window.React) instead of imports

var QuantileLossVisualizer = function QuantileLossVisualizer() {
  // Canvas and visualization parameters
  var canvasRef = useRef(null);
  var _useState = useState(0.5),
    _useState2 = _slicedToArray(_useState, 2),
    tau = _useState2[0],
    setTau = _useState2[1];
  var _useState3 = useState(0),
    _useState4 = _slicedToArray(_useState3, 2),
    highlightError = _useState4[0],
    setHighlightError = _useState4[1];
  var _useState5 = useState(null),
    _useState6 = _slicedToArray(_useState5, 2),
    hoverInfo = _useState6[0],
    setHoverInfo = _useState6[1];

  // Calculate quantile loss for a given error and tau
  var quantileLoss = function quantileLoss(error, quantile) {
    return error >= 0 ? quantile * error : (1 - quantile) * -error;
  };

  // Calculate derivative of quantile loss
  var quantileLossDerivative = function quantileLossDerivative(error, quantile) {
    // Note: The derivative is not defined at error = 0, but we approximate
    if (Math.abs(error) < 0.001) return 0;
    return error > 0 ? -quantile : 1 - quantile;
  };

  // Draw the visualization on canvas
  useEffect(function () {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var width = canvas.width;
    var height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    // Define visualization parameters
    var padding = {
      left: 60,
      right: 100,
      top: 60,
      bottom: 40
    };
    var plotWidth = width - padding.left - padding.right;
    var plotHeight = height / 2 - padding.top - padding.bottom / 2;

    // Error range from -5 to 5
    var errorMin = -5;
    var errorMax = 5;
    var errorRange = errorMax - errorMin;

    // X-axis scale (mapping error values to canvas coordinates)
    var xScale = plotWidth / errorRange;

    // Y-axis scales for loss and derivative
    var lossMax = 5;
    var lossScale = plotHeight / lossMax;
    var derivMin = -1;
    var derivMax = 1;
    var derivRange = derivMax - derivMin;
    var derivScale = plotHeight / derivRange;

    // Draw the loss function plot (top)
    // --------------------------------

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Y-axis
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);

    // X-axis
    ctx.moveTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Draw grid lines and labels for top plot
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;

    // Y-axis grid lines and labels
    var lossTickCount = 5;
    var lossTickStep = lossMax / lossTickCount;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';
    for (var i = 0; i <= lossTickCount; i++) {
      var yValue = i * lossTickStep;
      var y = padding.top + plotHeight - yValue * lossScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + plotWidth, y);
      ctx.stroke();

      // Label
      ctx.fillText(yValue.toFixed(1), padding.left - 10, y);
    }

    // X-axis grid lines and labels
    var errorTickValues = [-4, -2, 0, 2, 4];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    errorTickValues.forEach(function (error) {
      var x = padding.left + (error - errorMin) * xScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + plotHeight);
      ctx.stroke();

      // Label
      ctx.fillText(error.toString(), x, padding.top + plotHeight + 10);
    });

    // Draw different quantile loss functions
    var lossColors = {
      median: '#00C853',
      // Median (τ = 0.5)
      custom: '#FF6D00' // Custom τ value
    };

    // Function to draw a loss curve
    var drawLossCurve = function drawLossCurve(quantileValue, color, label, yOffset) {
      var lineWidth = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : 2;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      for (var x = 0; x <= plotWidth; x += 1) {
        var error = errorMin + x / plotWidth * errorRange;
        var loss = quantileLoss(error, quantileValue);

        // Ensure loss is within visible range
        var clampedLoss = Math.min(loss, lossMax);
        var canvasX = padding.left + x;
        var canvasY = padding.top + plotHeight - clampedLoss * lossScale;
        if (x === 0) {
          ctx.moveTo(canvasX, canvasY);
        } else {
          ctx.lineTo(canvasX, canvasY);
        }
      }
      ctx.stroke();

      // Add label to the curve
      ctx.font = '12px Arial';
      ctx.fillStyle = color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      // Position labels inside the plotting area to avoid cutoff
      ctx.fillText(label, padding.left + plotWidth - 120, padding.top + yOffset);
    };

    // Draw median quantile loss (τ = 0.5)
    drawLossCurve(0.5, lossColors.median, 'Median Regression', 20);

    // Draw custom quantile loss
    if (tau !== 0.5) {
      drawLossCurve(tau, lossColors.custom, "Quantile Regression", 40);
    }

    // Draw the derivative plot (bottom)
    // --------------------------------
    var derivPlotTop = padding.top * 2 + plotHeight;

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Y-axis
    ctx.moveTo(padding.left, derivPlotTop);
    ctx.lineTo(padding.left, derivPlotTop + plotHeight);

    // X-axis - centered at the middle of the plot height for derivative
    var xAxisY = derivPlotTop + plotHeight / 2;
    ctx.moveTo(padding.left, xAxisY);
    ctx.lineTo(padding.left + plotWidth, xAxisY);
    ctx.stroke();

    // Draw grid lines and labels for derivative plot
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;

    // Y-axis grid lines and labels
    var derivTickValues = [-1, -0.5, 0, 0.5, 1];
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';
    derivTickValues.forEach(function (tick) {
      var y = derivPlotTop + plotHeight / 2 - tick * derivScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + plotWidth, y);
      ctx.stroke();

      // Label
      ctx.fillText(tick.toFixed(1), padding.left - 10, y);
    });

    // X-axis grid lines reuse the same values as the top plot
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    errorTickValues.forEach(function (error) {
      var x = padding.left + (error - errorMin) * xScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(x, derivPlotTop);
      ctx.lineTo(x, derivPlotTop + plotHeight);
      ctx.stroke();

      // Label
      ctx.fillText(error.toString(), x, derivPlotTop + plotHeight + 10);
    });

    // Function to draw a derivative curve
    var drawDerivativeCurve = function drawDerivativeCurve(quantileValue, color) {
      var discontinuities = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : true;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (discontinuities) {
        // Draw with visible discontinuity at x = 0
        // Negative side
        ctx.beginPath();
        for (var x = 0; x < plotWidth / 2; x += 1) {
          var error = errorMin + x / plotWidth * errorRange;
          var deriv = quantileLossDerivative(error, quantileValue);
          var canvasX = padding.left + x;
          var canvasY = derivPlotTop + plotHeight / 2 - deriv * derivScale;
          if (x === 0) {
            ctx.moveTo(canvasX, canvasY);
          } else {
            ctx.lineTo(canvasX, canvasY);
          }
        }
        ctx.stroke();

        // Positive side
        ctx.beginPath();
        for (var _x = plotWidth / 2 + 1; _x <= plotWidth; _x += 1) {
          var _error = errorMin + _x / plotWidth * errorRange;
          var _deriv = quantileLossDerivative(_error, quantileValue);
          var _canvasX = padding.left + _x;
          var _canvasY = derivPlotTop + plotHeight / 2 - _deriv * derivScale;
          if (_x === plotWidth / 2 + 1) {
            ctx.moveTo(_canvasX, _canvasY);
          } else {
            ctx.lineTo(_canvasX, _canvasY);
          }
        }
        ctx.stroke();

        // Draw points at the discontinuity
        var midX = padding.left + plotWidth / 2;

        // Point at (0, -(1-τ))
        var y1 = derivPlotTop + plotHeight / 2 - -(1 - quantileValue) * derivScale;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(midX, y1, 4, 0, Math.PI * 2);
        ctx.fill();

        // Point at (0, -τ)
        var y2 = derivPlotTop + plotHeight / 2 - -quantileValue * derivScale;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(midX, y2, 4, 0, Math.PI * 2);
        ctx.fill();

        // Vertical line showing discontinuity
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(midX, y1);
        ctx.lineTo(midX, y2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    };

    // Draw derivative curves
    drawDerivativeCurve(0.5, lossColors.median); // Median

    if (tau !== 0.5) {
      drawDerivativeCurve(tau, lossColors.custom); // Custom τ
    }

    // Axis labels
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Derivative of Loss Function', padding.left + plotWidth / 2, derivPlotTop - 25);

    // Axis labels for both plots
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Error (r - r̂)', padding.left + plotWidth / 2, height - 15);
    ctx.save();
    ctx.translate(15, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(15, derivPlotTop + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Derivative', 0, 0);
    ctx.restore();

    // Draw title
    ctx.fillStyle = '#111';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = 'bold 16px Arial';
    ctx.fillText('Quantile Loss Function', padding.left + plotWidth / 2, 15);

    // Highlight specific error value if requested
    if (highlightError >= errorMin && highlightError <= errorMax) {
      var x = padding.left + (highlightError - errorMin) * xScale;

      // Vertical line
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + plotHeight);
      ctx.moveTo(x, derivPlotTop);
      ctx.lineTo(x, derivPlotTop + plotHeight);
      ctx.stroke();
      ctx.setLineDash([]);

      // Points for each loss function at this error
      var highlightPoints = [];
      var medianLoss = quantileLoss(highlightError, 0.5);
      var _y = padding.top + plotHeight - Math.min(medianLoss, lossMax) * lossScale;
      highlightPoints.push({
        color: lossColors.median,
        y: _y
      });
      if (tau !== 0.5) {
        var customLoss = quantileLoss(highlightError, tau);
        var _y2 = padding.top + plotHeight - Math.min(customLoss, lossMax) * lossScale;
        highlightPoints.push({
          color: lossColors.custom,
          y: _y2
        });
      }

      // Draw the points
      highlightPoints.forEach(function (point) {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw derivative points
      var derivPoints = [];
      var medianDeriv = quantileLossDerivative(highlightError, 0.5);
      var yDeriv = derivPlotTop + plotHeight / 2 - medianDeriv * derivScale;
      derivPoints.push({
        color: lossColors.median,
        y: yDeriv
      });
      if (tau !== 0.5) {
        var customDeriv = quantileLossDerivative(highlightError, tau);
        var _yDeriv = derivPlotTop + plotHeight / 2 - customDeriv * derivScale;
        derivPoints.push({
          color: lossColors.custom,
          y: _yDeriv
        });
      }

      // Draw the derivative points
      derivPoints.forEach(function (point) {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw hover information
    if (hoverInfo) {
      var _x2 = hoverInfo.x,
        _y3 = hoverInfo.y,
        error = hoverInfo.error,
        values = hoverInfo.values;

      // Background for tooltip
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.strokeStyle = '#aaa';
      ctx.lineWidth = 1;
      var tooltipWidth = 200;
      var tooltipHeight = 20 + values.length * 20;
      var tooltipX = Math.min(_x2 + 10, width - padding.right - tooltipWidth);
      var tooltipY = Math.min(_y3 + 10, height - padding.bottom - tooltipHeight);
      ctx.beginPath();
      ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 5);
      ctx.fill();
      ctx.stroke();

      // Tooltip content
      ctx.fillStyle = '#333';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.font = 'bold 12px Arial';
      ctx.fillText("Error: ".concat(error.toFixed(2)), tooltipX + 10, tooltipY + 10);
      ctx.font = '12px Arial';
      values.forEach(function (value, index) {
        var valueY = tooltipY + 30 + index * 20;
        ctx.fillStyle = value.color;
        ctx.fillText("".concat(value.label, ": ").concat(value.value.toFixed(4)), tooltipX + 10, valueY);
      });
    }

    // Draw legend
    var legendX = padding.left + plotWidth - 120;
    var legendY = padding.top + 30;
  }, [tau, highlightError, hoverInfo]);

  // Handle mouse movement over canvas for hover effects
  var handleMouseMove = function handleMouseMove(e) {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    var padding = {
      left: 60,
      right: 100,
      top: 60,
      bottom: 40
    };
    var plotWidth = canvas.width - padding.left - padding.right;
    var plotHeight = canvas.height / 2 - padding.top - padding.bottom / 2;

    // Check if mouse is in the plot area (either top or bottom)
    var isInTopPlot = x >= padding.left && x <= padding.left + plotWidth && y >= padding.top && y <= padding.top + plotHeight;
    var derivPlotTop = padding.top * 2 + plotHeight;
    var isInBottomPlot = x >= padding.left && x <= padding.left + plotWidth && y >= derivPlotTop && y <= derivPlotTop + plotHeight;
    if (isInTopPlot || isInBottomPlot) {
      // Calculate corresponding error value
      var errorMin = -5;
      var errorMax = 5;
      var errorRange = errorMax - errorMin;
      var error = errorMin + (x - padding.left) / plotWidth * errorRange;

      // Set highlight position
      setHighlightError(error);

      // Calculate loss and derivative values
      var values = [];

      // Median loss and derivative
      var medianLoss = quantileLoss(error, 0.5);
      values.push({
        label: 'Median Loss (τ = 0.5)',
        value: medianLoss,
        color: '#00C853'
      });
      var medianDeriv = quantileLossDerivative(error, 0.5);
      values.push({
        label: 'Median Derivative',
        value: medianDeriv,
        color: '#00C853'
      });

      // Custom quantile loss and derivative (if different from median)
      if (tau !== 0.5) {
        var customLoss = quantileLoss(error, tau);
        values.push({
          label: "Quantile Loss (\u03C4 = ".concat(tau.toFixed(2), ")"),
          value: customLoss,
          color: '#FF6D00'
        });
        var customDeriv = quantileLossDerivative(error, tau);
        values.push({
          label: "Quantile Derivative (\u03C4 = ".concat(tau.toFixed(2), ")"),
          value: customDeriv,
          color: '#FF6D00'
        });
      }
      setHoverInfo({
        x: x,
        y: y,
        error: error,
        values: values
      });
    } else {
      setHighlightError(null);
      setHoverInfo(null);
    }
  };
  var handleMouseLeave = function handleMouseLeave() {
    setHighlightError(null);
    setHoverInfo(null);
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "p-4 w-full flex flex-col items-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mb-4 w-full max-w-4xl"
  }, /*#__PURE__*/React.createElement("p", {
    className: "mb-2"
  }, "Quantile regression estimates a specific quantile of the conditional distribution rather than the mean. The parameter \u03C4 \u2208 (0,1) determines which quantile is targeted (e.g., \u03C4 = 0.5 for median, \u03C4 = 0.9 for 90th percentile)."), /*#__PURE__*/React.createElement("p", {
    className: "mb-2"
  }, /*#__PURE__*/React.createElement("strong", null, "The loss function has an asymmetric weighting based on \u03C4:")), /*#__PURE__*/React.createElement("ul", {
    className: "list-disc ml-8 mb-2"
  }, /*#__PURE__*/React.createElement("li", null, "When prediction is too high (error ", '<', " 0): weighted by (1-\u03C4)"), /*#__PURE__*/React.createElement("li", null, "When prediction is too low (error ", '>', " 0): weighted by \u03C4"))), /*#__PURE__*/React.createElement("div", {
    className: "w-full max-w-4xl mb-4 flex items-center"
  }, /*#__PURE__*/React.createElement("label", {
    htmlFor: "tau",
    className: "font-medium mr-2"
  }, "Quantile (\u03C4):"), /*#__PURE__*/React.createElement("input", {
    id: "tau",
    type: "range",
    min: "0.01",
    max: "0.99",
    step: "0.01",
    value: tau,
    onChange: function onChange(e) {
      return setTau(parseFloat(e.target.value));
    },
    className: "w-60 mr-2"
  }), /*#__PURE__*/React.createElement("span", {
    className: "w-16 text-center font-mono"
  }, tau.toFixed(2))), /*#__PURE__*/React.createElement("canvas", {
    ref: canvasRef,
    width: 800,
    height: 600,
    className: "border border-gray-300 bg-white",
    onMouseMove: handleMouseMove,
    onMouseLeave: handleMouseLeave
  }), /*#__PURE__*/React.createElement("div", {
    className: "mt-6 w-full max-w-4xl text-sm"
  }, /*#__PURE__*/React.createElement("h3", {
    className: "font-bold text-lg mb-2"
  }, "Key Properties of Quantile Loss:"), /*#__PURE__*/React.createElement("ul", {
    className: "list-disc ml-6 space-y-2"
  }, /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Asymmetric Weighting:"), " The loss function penalizes overestimation and underestimation differently based on the value of \u03C4."), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Discontinuous Derivative:"), " The derivative has a discontinuity at zero, which is a key feature that makes the loss function pull the predictions toward the desired quantile."), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "\u03C4 = 0.5 (Median):"), " This special case gives equal weight to positive and negative errors, resulting in median regression (equivalent to minimizing Mean Absolute Error)."), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Risk-Averse Models:"), " High values of \u03C4 (e.g., 0.9) are especially useful for risk-averse predictions, where overestimation is more costly than underestimation."))));
};

/* export removed */ /* QuantileLossVisualizer will be assigned to window.QuantileLossVisualizer */
// Global export
if (typeof window !== 'undefined') {
  window.QuantileLossVisualizer = QuantileLossVisualizer;
}


    // Make sure the component is exported to the global window object
    if (typeof window !== 'undefined' && typeof QuantileLossVisualizer !== 'undefined') {
        window.QuantileLossVisualizer = QuantileLossVisualizer;
        console.log('Successfully exported QuantileLossVisualizer to global scope');
    }
})();
