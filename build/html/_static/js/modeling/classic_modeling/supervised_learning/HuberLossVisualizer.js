
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

var _HuberLossVisualizer = function HuberLossVisualizer() {
  // Canvas and visualization parameters
  var canvasRef = useRef(null);
  var _useState = useState(1.0),
    _useState2 = _slicedToArray(_useState, 2),
    delta = _useState2[0],
    setDelta = _useState2[1];
  var _useState3 = useState(true),
    _useState4 = _slicedToArray(_useState3, 2),
    showLogCosh = _useState4[0],
    setShowLogCosh = _useState4[1];
  var _useState5 = useState(false),
    _useState6 = _slicedToArray(_useState5, 2),
    showMSE = _useState6[0],
    setShowMSE = _useState6[1];
  var _useState7 = useState(false),
    _useState8 = _slicedToArray(_useState7, 2),
    showMAE = _useState8[0],
    setShowMAE = _useState8[1];
  var _useState9 = useState(0),
    _useState10 = _slicedToArray(_useState9, 2),
    highlightError = _useState10[0],
    setHighlightError = _useState10[1];
  var _useState11 = useState(null),
    _useState12 = _slicedToArray(_useState11, 2),
    hoverInfo = _useState12[0],
    setHoverInfo = _useState12[1];

  // Huber loss function
  var huberLoss = function huberLoss(a, delta) {
    return Math.abs(a) <= delta ? 0.5 * a * a : delta * (Math.abs(a) - 0.5 * delta);
  };

  // Export as default and also make available globally
  /* export removed */ /* HuberLossVisualizer will be assigned to window.HuberLossVisualizer */

  // Make component available globally
  if (typeof window !== 'undefined') {
    window.HuberLossVisualizer = _HuberLossVisualizer;
  }

  // Log-Cosh loss function
  var logCoshLoss = function logCoshLoss(a) {
    return Math.log(Math.cosh(a));
  };

  // Derivative of Huber loss
  var huberDerivative = function huberDerivative(a, delta) {
    return Math.abs(a) <= delta ? a : delta * Math.sign(a);
  };

  // Derivative of Log-Cosh loss
  var logCoshDerivative = function logCoshDerivative(a) {
    return Math.tanh(a);
  };

  // MSE loss
  var mseLoss = function mseLoss(a) {
    return 0.5 * a * a;
  };

  // MAE loss
  var maeLoss = function maeLoss(a) {
    return Math.abs(a);
  };

  // Draw the visualization on canvas
  useEffect(function () {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var width = canvas.width;
    var height = canvas.height / 2;
    var padding = {
      left: 60,
      right: 170,
      top: 50,
      bottom: 50
    };

    // Clear canvas
    ctx.clearRect(0, 0, width, height * 2);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height * 2);

    // Define plot dimensions
    var plotWidth = width - padding.left - padding.right;
    var plotHeight = height - padding.top - padding.bottom;

    // Error range from -5 to 5
    var errorMin = -5;
    var errorMax = 5;
    var errorRange = errorMax - errorMin;

    // X-axis scale
    var xScale = plotWidth / errorRange;

    // Y-axis scales for loss and derivative
    var lossMax = 5;
    var lossScale = plotHeight / lossMax;
    var derivMin = -1.2;
    var derivMax = 1.2;
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

    // Define loss function colors
    var lossColors = {
      huber: '#4682B4',
      // Steel Blue
      logCosh: '#9C27B0',
      // Purple
      mse: '#D2691E',
      // Chocolate
      mae: '#228B22' // Forest Green
    };

    // Function to draw a loss curve with clipping to plot boundaries
    var drawLossCurve = function drawLossCurve(lossFunc, color) {
      var lineWidth = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 2;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      var started = false;
      for (var x = 0; x <= plotWidth; x += 1) {
        var error = errorMin + x / plotWidth * errorRange;
        var loss = lossFunc(error);

        // If loss is out of the visible range, break the path
        if (loss < 0 || loss > lossMax) {
          if (started) {
            ctx.stroke();
            ctx.beginPath();
            started = false;
          }
          continue;
        }

        // Compute Y from the *actual* loss, not the clamped loss
        var canvasX = padding.left + x;
        var canvasY = padding.top + plotHeight - loss * lossScale;
        if (!started) {
          ctx.moveTo(canvasX, canvasY);
          started = true;
        } else {
          ctx.lineTo(canvasX, canvasY);
        }
      }
      if (started) {
        ctx.stroke();
      }
    };

    // Draw MSE loss if enabled
    if (showMSE) {
      drawLossCurve(function (a) {
        return mseLoss(a);
      }, lossColors.mse, 1.5);
    }

    // Draw MAE loss if enabled
    if (showMAE) {
      drawLossCurve(function (a) {
        return maeLoss(a);
      }, lossColors.mae, 1.5);
    }

    // Draw Huber loss (always shown)
    drawLossCurve(function (a) {
      return huberLoss(a, delta);
    }, lossColors.huber, 2);

    // Draw Log-Cosh loss if enabled
    if (showLogCosh) {
      drawLossCurve(function (a) {
        return logCoshLoss(a);
      }, lossColors.logCosh, 2);
    }

    // Draw vertical lines at delta and -delta
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    var deltaX = padding.left + (delta - errorMin) * xScale;
    var negDeltaX = padding.left + (-delta - errorMin) * xScale;
    ctx.moveTo(deltaX, padding.top);
    ctx.lineTo(deltaX, padding.top + plotHeight);
    ctx.moveTo(negDeltaX, padding.top);
    ctx.lineTo(negDeltaX, padding.top + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw the derivative plot (bottom)
    // --------------------------------
    var derivPlotTop = height + padding.top;

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Y-axis
    ctx.moveTo(padding.left, derivPlotTop);
    ctx.lineTo(padding.left, derivPlotTop + plotHeight);

    // X-axis - at the center of the derivative plot
    ctx.moveTo(padding.left, derivPlotTop + plotHeight / 2);
    ctx.lineTo(padding.left + plotWidth, derivPlotTop + plotHeight / 2);
    ctx.stroke();

    // Draw grid lines and labels for derivative plot
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;

    // Y-axis grid lines and labels
    var derivTickValues = [-1, -0.5, 0, 0.5, 1];
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
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

    // X-axis grid lines and labels
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

    // Function to draw a derivative curve with clipping
    var drawDerivativeCurve = function drawDerivativeCurve(derivFunc, color) {
      var smoothCurve = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
      var lineWidth = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : 2;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      var derivPlotBottom = derivPlotTop + plotHeight;
      var derivMidY = derivPlotTop + plotHeight / 2;
      if (smoothCurve) {
        // Draw a continuous smooth curve
        ctx.beginPath();
        var started = false;
        for (var x = 0; x <= plotWidth; x += 1) {
          var error = errorMin + x / plotWidth * errorRange;
          var deriv = derivFunc(error);
          var canvasX = padding.left + x;
          var canvasY = derivMidY - deriv * derivScale;

          // Only draw if within plot boundaries
          if (canvasY >= derivPlotTop && canvasY <= derivPlotBottom) {
            if (!started) {
              ctx.moveTo(canvasX, canvasY);
              started = true;
            } else {
              ctx.lineTo(canvasX, canvasY);
            }
          } else if (started) {
            // If we've gone outside but were inside before, calculate intersection
            var prevError = errorMin + (x - 1) / plotWidth * errorRange;
            var prevDeriv = derivFunc(prevError);
            var prevY = derivMidY - prevDeriv * derivScale;
            if (prevY >= derivPlotTop && prevY <= derivPlotBottom) {
              // Linear interpolation to find boundary intersection
              if (canvasY < derivPlotTop) {
                // Intersection with top boundary
                var t = (derivPlotTop - prevY) / (canvasY - prevY);
                var intersectX = padding.left + (x - 1) + t;
                ctx.lineTo(intersectX, derivPlotTop);
              } else {
                // Intersection with bottom boundary
                var _t = (derivPlotBottom - prevY) / (canvasY - prevY);
                var _intersectX = padding.left + (x - 1) + _t;
                ctx.lineTo(_intersectX, derivPlotBottom);
              }
            }
            ctx.stroke();
            ctx.beginPath();
            started = false;
          } else if (x > 0) {
            // Check if we're crossing back into the plot
            var _prevError = errorMin + (x - 1) / plotWidth * errorRange;
            var _prevDeriv = derivFunc(_prevError);
            var _prevY = derivMidY - _prevDeriv * derivScale;
            if (_prevY < derivPlotTop && canvasY >= derivPlotTop || _prevY > derivPlotBottom && canvasY <= derivPlotBottom) {
              // Calculate intersection and continue
              if (_prevY < derivPlotTop) {
                // Coming back in through top
                var _t2 = (derivPlotTop - _prevY) / (canvasY - _prevY);
                var _intersectX2 = padding.left + (x - 1) + _t2;
                ctx.moveTo(_intersectX2, derivPlotTop);
              } else {
                // Coming back in through bottom
                var _t3 = (derivPlotBottom - _prevY) / (canvasY - _prevY);
                var _intersectX3 = padding.left + (x - 1) + _t3;
                ctx.moveTo(_intersectX3, derivPlotBottom);
              }
              ctx.lineTo(canvasX, canvasY);
              started = true;
            }
          }
        }
        if (started) {
          ctx.stroke();
        }
      } else {
        // Draw with discontinuity at zero (for Huber loss or MAE)
        // Left side (negative)
        ctx.beginPath();
        var _started = false;
        for (var _x = 0; _x < plotWidth / 2; _x += 1) {
          var _error = errorMin + _x / plotWidth * errorRange;
          var _deriv = derivFunc(_error);
          var _canvasX = padding.left + _x;
          var _canvasY = derivMidY - _deriv * derivScale;
          if (_canvasY >= derivPlotTop && _canvasY <= derivPlotBottom) {
            if (!_started) {
              ctx.moveTo(_canvasX, _canvasY);
              _started = true;
            } else {
              ctx.lineTo(_canvasX, _canvasY);
            }
          } else if (_started) {
            // Handle boundary crossing
            var _prevError2 = errorMin + (_x - 1) / plotWidth * errorRange;
            var _prevDeriv2 = derivFunc(_prevError2);
            var _prevY2 = derivMidY - _prevDeriv2 * derivScale;
            if (_prevY2 >= derivPlotTop && _prevY2 <= derivPlotBottom) {
              // Calculate intersection
              if (_canvasY < derivPlotTop) {
                var _t4 = (derivPlotTop - _prevY2) / (_canvasY - _prevY2);
                var _intersectX4 = padding.left + (_x - 1) + _t4;
                ctx.lineTo(_intersectX4, derivPlotTop);
              } else {
                var _t5 = (derivPlotBottom - _prevY2) / (_canvasY - _prevY2);
                var _intersectX5 = padding.left + (_x - 1) + _t5;
                ctx.lineTo(_intersectX5, derivPlotBottom);
              }
            }
            ctx.stroke();
            ctx.beginPath();
            _started = false;
          }
        }
        if (_started) {
          ctx.stroke();
        }

        // Right side (positive)
        ctx.beginPath();
        _started = false;
        for (var _x2 = plotWidth / 2; _x2 <= plotWidth; _x2 += 1) {
          var _error2 = errorMin + _x2 / plotWidth * errorRange;
          var _deriv2 = derivFunc(_error2);
          var _canvasX2 = padding.left + _x2;
          var _canvasY2 = derivMidY - _deriv2 * derivScale;
          if (_canvasY2 >= derivPlotTop && _canvasY2 <= derivPlotBottom) {
            if (!_started) {
              ctx.moveTo(_canvasX2, _canvasY2);
              _started = true;
            } else {
              ctx.lineTo(_canvasX2, _canvasY2);
            }
          } else if (_started) {
            // Handle boundary crossing
            var _prevError3 = errorMin + (_x2 - 1) / plotWidth * errorRange;
            var _prevDeriv3 = derivFunc(_prevError3);
            var _prevY3 = derivMidY - _prevDeriv3 * derivScale;
            if (_prevY3 >= derivPlotTop && _prevY3 <= derivPlotBottom) {
              // Calculate intersection
              if (_canvasY2 < derivPlotTop) {
                var _t6 = (derivPlotTop - _prevY3) / (_canvasY2 - _prevY3);
                var _intersectX6 = padding.left + (_x2 - 1) + _t6;
                ctx.lineTo(_intersectX6, derivPlotTop);
              } else {
                var _t7 = (derivPlotBottom - _prevY3) / (_canvasY2 - _prevY3);
                var _intersectX7 = padding.left + (_x2 - 1) + _t7;
                ctx.lineTo(_intersectX7, derivPlotBottom);
              }
            }
            ctx.stroke();
            ctx.beginPath();
            _started = false;
          }
        }
        if (_started) {
          ctx.stroke();
        }

        // Draw discontinuity markers
        var midX = padding.left + plotWidth / 2;
        var leftDeriv = derivFunc(-0.001);
        var rightDeriv = derivFunc(0.001);
        var leftY = derivMidY - leftDeriv * derivScale;
        var rightY = derivMidY - rightDeriv * derivScale;

        // Points at discontinuity
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(midX, leftY, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(midX, rightY, 4, 0, Math.PI * 2);
        ctx.fill();

        // Vertical line showing discontinuity
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(midX, leftY);
        ctx.lineTo(midX, rightY);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    };

    // Draw MSE derivative if enabled
    if (showMSE) {
      drawDerivativeCurve(function (a) {
        return a;
      }, lossColors.mse, true, 1.5);
    }

    // Draw MAE derivative if enabled
    if (showMAE) {
      drawDerivativeCurve(function (a) {
        return Math.sign(a);
      }, lossColors.mae, false, 1.5);
    }

    // Draw Huber derivative (always shown)
    drawDerivativeCurve(function (a) {
      return huberDerivative(a, delta);
    }, lossColors.huber, false, 2);

    // Draw Log-Cosh derivative if enabled
    if (showLogCosh) {
      drawDerivativeCurve(function (a) {
        return logCoshDerivative(a);
      }, lossColors.logCosh, true, 2);
    }

    // Draw vertical lines at delta and -delta for derivative
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(deltaX, derivPlotTop);
    ctx.lineTo(deltaX, derivPlotTop + plotHeight);
    ctx.moveTo(negDeltaX, derivPlotTop);
    ctx.lineTo(negDeltaX, derivPlotTop + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    // Add labels and titles
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Plot titles
    ctx.fillText('Loss Functions', padding.left + plotWidth / 2, padding.top - 30);
    ctx.fillText('Derivatives', padding.left + plotWidth / 2, derivPlotTop - 30);

    // Axis labels
    ctx.fillText('Error (r - r̂)', padding.left + plotWidth / 2, derivPlotTop + plotHeight + 30);
    ctx.save();
    ctx.translate(padding.left - 40, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(padding.left - 40, derivPlotTop + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Derivative', 0, 0);
    ctx.restore();

    // Draw the legend
    var legendX = padding.left + plotWidth + 10;
    var legendY = padding.top + 20;
    var legendSpacing = 25;
    var legendItems = [];

    // Always add Huber loss to legend
    legendItems.push({
      color: lossColors.huber,
      label: "Huber Loss (\u03B4 = ".concat(delta.toFixed(1), ")")
    });

    // Add Log-Cosh if enabled
    if (showLogCosh) {
      legendItems.push({
        color: lossColors.logCosh,
        label: 'Log-Cosh Loss'
      });
    }

    // Add MSE if enabled
    if (showMSE) {
      legendItems.push({
        color: lossColors.mse,
        label: 'MSE'
      });
    }

    // Add MAE if enabled
    if (showMAE) {
      legendItems.push({
        color: lossColors.mae,
        label: 'MAE'
      });
    }

    // Draw the legend items
    legendItems.forEach(function (item, index) {
      var y = legendY + index * legendSpacing;

      // Line
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(legendX, y);
      ctx.lineTo(legendX + 30, y);
      ctx.stroke();

      // Label
      ctx.fillStyle = item.color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.font = '12px Arial';
      ctx.fillText(item.label, legendX + 40, y);
    });

    // Highlight specific error value if requested
    if (highlightError >= errorMin && highlightError <= errorMax) {
      var x = padding.left + (highlightError - errorMin) * xScale;

      // Vertical line through both plots
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

      // Draw points on the loss curves
      var lossPoints = [];

      // Huber loss point
      var huberValue = huberLoss(highlightError, delta);
      var huberY = padding.top + plotHeight - Math.min(huberValue, lossMax) * lossScale;
      if (huberY >= padding.top && huberY <= padding.top + plotHeight) {
        lossPoints.push({
          color: lossColors.huber,
          y: huberY
        });
      }

      // Log-Cosh loss point
      if (showLogCosh) {
        var logCoshValue = logCoshLoss(highlightError);
        var logCoshY = padding.top + plotHeight - Math.min(logCoshValue, lossMax) * lossScale;
        if (logCoshY >= padding.top && logCoshY <= padding.top + plotHeight) {
          lossPoints.push({
            color: lossColors.logCosh,
            y: logCoshY
          });
        }
      }

      // MSE loss point
      if (showMSE) {
        var mseValue = mseLoss(highlightError);
        var mseY = padding.top + plotHeight - Math.min(mseValue, lossMax) * lossScale;
        if (mseY >= padding.top && mseY <= padding.top + plotHeight) {
          lossPoints.push({
            color: lossColors.mse,
            y: mseY
          });
        }
      }

      // MAE loss point
      if (showMAE) {
        var maeValue = maeLoss(highlightError);
        var maeY = padding.top + plotHeight - Math.min(maeValue, lossMax) * lossScale;
        if (maeY >= padding.top && maeY <= padding.top + plotHeight) {
          lossPoints.push({
            color: lossColors.mae,
            y: maeY
          });
        }
      }

      // Draw the points
      lossPoints.forEach(function (point) {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw points on the derivative curves
      var derivPoints = [];
      var derivMidY = derivPlotTop + plotHeight / 2;

      // Huber derivative point
      var huberDerivValue = huberDerivative(highlightError, delta);
      var huberDerivY = derivMidY - huberDerivValue * derivScale;
      if (huberDerivY >= derivPlotTop && huberDerivY <= derivPlotTop + plotHeight) {
        derivPoints.push({
          color: lossColors.huber,
          y: huberDerivY
        });
      }

      // Log-Cosh derivative point
      if (showLogCosh) {
        var logCoshDerivValue = logCoshDerivative(highlightError);
        var logCoshDerivY = derivMidY - logCoshDerivValue * derivScale;
        if (logCoshDerivY >= derivPlotTop && logCoshDerivY <= derivPlotTop + plotHeight) {
          derivPoints.push({
            color: lossColors.logCosh,
            y: logCoshDerivY
          });
        }
      }

      // MSE derivative point
      if (showMSE) {
        var mseDerivValue = highlightError;
        var mseDerivY = derivMidY - mseDerivValue * derivScale;
        if (mseDerivY >= derivPlotTop && mseDerivY <= derivPlotTop + plotHeight) {
          derivPoints.push({
            color: lossColors.mse,
            y: mseDerivY
          });
        }
      }

      // MAE derivative point
      if (showMAE) {
        var maeDerivValue = Math.sign(highlightError);
        var maeDerivY = derivMidY - maeDerivValue * derivScale;
        if (maeDerivY >= derivPlotTop && maeDerivY <= derivPlotTop + plotHeight) {
          derivPoints.push({
            color: lossColors.mae,
            y: maeDerivY
          });
        }
      }

      // Draw the derivative points
      derivPoints.forEach(function (point) {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw hover information
    if (hoverInfo) {
      var _x3 = hoverInfo.x,
        _y = hoverInfo.y,
        error = hoverInfo.error,
        values = hoverInfo.values;

      // Background for tooltip
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.strokeStyle = '#aaa';
      ctx.lineWidth = 1;
      var tooltipWidth = 200;
      var tooltipHeight = 20 + values.length * 20;
      var tooltipX = Math.min(_x3 + 10, width - padding.right - tooltipWidth);
      var tooltipY = Math.min(_y + 10, height * 2 - 40 - tooltipHeight);
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
  }, [delta, showLogCosh, showMSE, showMAE, highlightError, hoverInfo]);

  // Handle mouse movement over canvas for hover effects
  var handleMouseMove = function handleMouseMove(e) {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    var padding = {
      left: 60,
      right: 60,
      top: 50,
      bottom: 40
    };
    var height = canvas.height / 2;
    var plotWidth = canvas.width - padding.left - padding.right;
    var plotHeight = height - padding.top - padding.bottom;

    // Check if mouse is in the plot area (either top or bottom)
    var isInTopPlot = x >= padding.left && x <= padding.left + plotWidth && y >= padding.top && y <= padding.top + plotHeight;
    var derivPlotTop = height + padding.top;
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

      // Huber loss
      var huberValue = huberLoss(error, delta);
      values.push({
        label: "Huber Loss (\u03B4 = ".concat(delta.toFixed(1), ")"),
        value: huberValue,
        color: '#4682B4'
      });
      var huberDerivValue = huberDerivative(error, delta);
      values.push({
        label: 'Huber Derivative',
        value: huberDerivValue,
        color: '#4682B4'
      });

      // Log-Cosh loss
      if (showLogCosh) {
        var logCoshValue = logCoshLoss(error);
        values.push({
          label: 'Log-Cosh Loss',
          value: logCoshValue,
          color: '#9C27B0'
        });
        var logCoshDerivValue = logCoshDerivative(error);
        values.push({
          label: 'Log-Cosh Derivative',
          value: logCoshDerivValue,
          color: '#9C27B0'
        });
      }

      // MSE loss
      if (showMSE) {
        var mseValue = mseLoss(error);
        values.push({
          label: 'MSE Loss',
          value: mseValue,
          color: '#D2691E'
        });
        values.push({
          label: 'MSE Derivative',
          value: error,
          color: '#D2691E'
        });
      }

      // MAE loss
      if (showMAE) {
        var maeValue = maeLoss(error);
        values.push({
          label: 'MAE Loss',
          value: maeValue,
          color: '#228B22'
        });
        values.push({
          label: 'MAE Derivative',
          value: Math.sign(error),
          color: '#228B22'
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
  }, /*#__PURE__*/React.createElement("h2", {
    className: "text-xl font-bold mb-4"
  }, "MAE vs MSE vs Huber vs Log-Cosh Loss Comparison"), /*#__PURE__*/React.createElement("div", {
    className: "w-full max-w-4xl mb-4 flex items-center gap-4 justify-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center space-x-2"
  }, /*#__PURE__*/React.createElement("label", {
    htmlFor: "delta",
    className: "font-medium"
  }, "Huber \u03B4:"), /*#__PURE__*/React.createElement("input", {
    id: "delta",
    type: "range",
    min: "0.1",
    max: "2.0",
    step: "0.1",
    value: delta,
    onChange: function onChange(e) {
      return setDelta(parseFloat(e.target.value));
    },
    className: "w-40"
  }), /*#__PURE__*/React.createElement("span", {
    className: "w-10 text-center"
  }, delta.toFixed(1))), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center space-x-2"
  }, /*#__PURE__*/React.createElement("label", {
    className: "inline-flex items-center"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: showLogCosh,
    onChange: function onChange(e) {
      return setShowLogCosh(e.target.checked);
    },
    className: "mr-1"
  }), "Log-Cosh"), /*#__PURE__*/React.createElement("label", {
    className: "inline-flex items-center"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: showMSE,
    onChange: function onChange(e) {
      return setShowMSE(e.target.checked);
    },
    className: "mr-1"
  }), "MSE"), /*#__PURE__*/React.createElement("label", {
    className: "inline-flex items-center"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: showMAE,
    onChange: function onChange(e) {
      return setShowMAE(e.target.checked);
    },
    className: "mr-1"
  }), "MAE"))), /*#__PURE__*/React.createElement("canvas", {
    ref: canvasRef,
    width: 800,
    height: 800,
    className: "border border-gray-300 bg-white",
    onMouseMove: handleMouseMove,
    onMouseLeave: handleMouseLeave
  }));
};
// Global export
if (typeof window !== 'undefined') {
  window.HuberLossVisualizer = _HuberLossVisualizer;
}


    // Make sure the component is exported to the global window object
    if (typeof window !== 'undefined' && typeof HuberLossVisualizer !== 'undefined') {
        window.HuberLossVisualizer = HuberLossVisualizer;
        console.log('Successfully exported HuberLossVisualizer to global scope');
    }
})();
