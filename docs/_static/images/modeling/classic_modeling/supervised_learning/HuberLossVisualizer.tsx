import React, { useState, useEffect, useRef } from 'react';

const HuberLossVisualizer = () => {
  // Canvas and visualization parameters
  const canvasRef = useRef(null);
  const [delta, setDelta] = useState(1.0);
  const [showLogCosh, setShowLogCosh] = useState(true);
  const [showMSE, setShowMSE] = useState(false);
  const [showMAE, setShowMAE] = useState(false);
  const [highlightError, setHighlightError] = useState(0);
  const [hoverInfo, setHoverInfo] = useState(null);

  // Huber loss function
  const huberLoss = (a, delta) => {
    return Math.abs(a) <= delta
      ? 0.5 * a * a
      : delta * (Math.abs(a) - 0.5 * delta);
  };

// Export as default and also make available globally
export default HuberLossVisualizer;

// Make component available globally
if (typeof window !== 'undefined') {
  window.HuberLossVisualizer = HuberLossVisualizer;
}

  // Log-Cosh loss function
  const logCoshLoss = (a) => {
    return Math.log(Math.cosh(a));
  };

  // Derivative of Huber loss
  const huberDerivative = (a, delta) => {
    return Math.abs(a) <= delta
      ? a
      : delta * Math.sign(a);
  };

  // Derivative of Log-Cosh loss
  const logCoshDerivative = (a) => {
    return Math.tanh(a);
  };

  // MSE loss
  const mseLoss = (a) => 0.5 * a * a;

  // MAE loss
  const maeLoss = (a) => Math.abs(a);

  // Draw the visualization on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height / 2;
    const padding = { left: 60, right: 170, top: 50, bottom: 50 };

    // Clear canvas
    ctx.clearRect(0, 0, width, height * 2);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height * 2);

    // Define plot dimensions
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Error range from -5 to 5
    const errorMin = -5;
    const errorMax = 5;
    const errorRange = errorMax - errorMin;

    // X-axis scale
    const xScale = plotWidth / errorRange;

    // Y-axis scales for loss and derivative
    const lossMax = 5;
    const lossScale = plotHeight / lossMax;

    const derivMin = -1.2;
    const derivMax = 1.2;
    const derivRange = derivMax - derivMin;
    const derivScale = plotHeight / derivRange;

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
    const lossTickCount = 5;
    const lossTickStep = lossMax / lossTickCount;

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';

    for (let i = 0; i <= lossTickCount; i++) {
      const yValue = i * lossTickStep;
      const y = padding.top + plotHeight - yValue * lossScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + plotWidth, y);
      ctx.stroke();

      // Label
      ctx.fillText(yValue.toFixed(1), padding.left - 10, y);
    }

    // X-axis grid lines and labels
    const errorTickValues = [-4, -2, 0, 2, 4];

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    errorTickValues.forEach(error => {
      const x = padding.left + (error - errorMin) * xScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + plotHeight);
      ctx.stroke();

      // Label
      ctx.fillText(error.toString(), x, padding.top + plotHeight + 10);
    });

    // Define loss function colors
    const lossColors = {
      huber: '#4682B4',  // Steel Blue
      logCosh: '#9C27B0', // Purple
      mse: '#D2691E',     // Chocolate
      mae: '#228B22'      // Forest Green
    };

    // Function to draw a loss curve with clipping to plot boundaries
    const drawLossCurve = (lossFunc, color, lineWidth = 2) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      let started = false;

      for (let x = 0; x <= plotWidth; x += 1) {
        const error = errorMin + (x / plotWidth) * errorRange;
        const loss = lossFunc(error);

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
        const canvasX = padding.left + x;
        const canvasY = padding.top + plotHeight - loss * lossScale;

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
      drawLossCurve((a) => mseLoss(a), lossColors.mse, 1.5);
    }

    // Draw MAE loss if enabled
    if (showMAE) {
      drawLossCurve((a) => maeLoss(a), lossColors.mae, 1.5);
    }

    // Draw Huber loss (always shown)
    drawLossCurve((a) => huberLoss(a, delta), lossColors.huber, 2);

    // Draw Log-Cosh loss if enabled
    if (showLogCosh) {
      drawLossCurve((a) => logCoshLoss(a), lossColors.logCosh, 2);
    }

    // Draw vertical lines at delta and -delta
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();

    const deltaX = padding.left + (delta - errorMin) * xScale;
    const negDeltaX = padding.left + (-delta - errorMin) * xScale;

    ctx.moveTo(deltaX, padding.top);
    ctx.lineTo(deltaX, padding.top + plotHeight);

    ctx.moveTo(negDeltaX, padding.top);
    ctx.lineTo(negDeltaX, padding.top + plotHeight);

    ctx.stroke();
    ctx.setLineDash([]);

    // Draw the derivative plot (bottom)
    // --------------------------------
    const derivPlotTop = height + padding.top;

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
    const derivTickValues = [-1, -0.5, 0, 0.5, 1];

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    derivTickValues.forEach(tick => {
      const y = derivPlotTop + plotHeight / 2 - tick * derivScale;

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

    errorTickValues.forEach(error => {
      const x = padding.left + (error - errorMin) * xScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(x, derivPlotTop);
      ctx.lineTo(x, derivPlotTop + plotHeight);
      ctx.stroke();

      // Label
      ctx.fillText(error.toString(), x, derivPlotTop + plotHeight + 10);
    });

    // Function to draw a derivative curve with clipping
    const drawDerivativeCurve = (derivFunc, color, smoothCurve = false, lineWidth = 2) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;

      const derivPlotBottom = derivPlotTop + plotHeight;
      const derivMidY = derivPlotTop + plotHeight / 2;

      if (smoothCurve) {
        // Draw a continuous smooth curve
        ctx.beginPath();
        let started = false;

        for (let x = 0; x <= plotWidth; x += 1) {
          const error = errorMin + (x / plotWidth) * errorRange;
          const deriv = derivFunc(error);
          const canvasX = padding.left + x;
          const canvasY = derivMidY - deriv * derivScale;

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
            const prevError = errorMin + ((x - 1) / plotWidth) * errorRange;
            const prevDeriv = derivFunc(prevError);
            const prevY = derivMidY - prevDeriv * derivScale;

            if (prevY >= derivPlotTop && prevY <= derivPlotBottom) {
              // Linear interpolation to find boundary intersection
              if (canvasY < derivPlotTop) {
                // Intersection with top boundary
                const t = (derivPlotTop - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.lineTo(intersectX, derivPlotTop);
              } else {
                // Intersection with bottom boundary
                const t = (derivPlotBottom - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.lineTo(intersectX, derivPlotBottom);
              }
            }

            ctx.stroke();
            ctx.beginPath();
            started = false;
          } else if (x > 0) {
            // Check if we're crossing back into the plot
            const prevError = errorMin + ((x - 1) / plotWidth) * errorRange;
            const prevDeriv = derivFunc(prevError);
            const prevY = derivMidY - prevDeriv * derivScale;

            if ((prevY < derivPlotTop && canvasY >= derivPlotTop) ||
                (prevY > derivPlotBottom && canvasY <= derivPlotBottom)) {
              // Calculate intersection and continue
              if (prevY < derivPlotTop) {
                // Coming back in through top
                const t = (derivPlotTop - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.moveTo(intersectX, derivPlotTop);
              } else {
                // Coming back in through bottom
                const t = (derivPlotBottom - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.moveTo(intersectX, derivPlotBottom);
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
        let started = false;

        for (let x = 0; x < plotWidth / 2; x += 1) {
          const error = errorMin + (x / plotWidth) * errorRange;
          const deriv = derivFunc(error);
          const canvasX = padding.left + x;
          const canvasY = derivMidY - deriv * derivScale;

          if (canvasY >= derivPlotTop && canvasY <= derivPlotBottom) {
            if (!started) {
              ctx.moveTo(canvasX, canvasY);
              started = true;
            } else {
              ctx.lineTo(canvasX, canvasY);
            }
          } else if (started) {
            // Handle boundary crossing
            const prevError = errorMin + ((x - 1) / plotWidth) * errorRange;
            const prevDeriv = derivFunc(prevError);
            const prevY = derivMidY - prevDeriv * derivScale;

            if (prevY >= derivPlotTop && prevY <= derivPlotBottom) {
              // Calculate intersection
              if (canvasY < derivPlotTop) {
                const t = (derivPlotTop - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.lineTo(intersectX, derivPlotTop);
              } else {
                const t = (derivPlotBottom - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.lineTo(intersectX, derivPlotBottom);
              }
            }

            ctx.stroke();
            ctx.beginPath();
            started = false;
          }
        }

        if (started) {
          ctx.stroke();
        }

        // Right side (positive)
        ctx.beginPath();
        started = false;

        for (let x = plotWidth / 2; x <= plotWidth; x += 1) {
          const error = errorMin + (x / plotWidth) * errorRange;
          const deriv = derivFunc(error);
          const canvasX = padding.left + x;
          const canvasY = derivMidY - deriv * derivScale;

          if (canvasY >= derivPlotTop && canvasY <= derivPlotBottom) {
            if (!started) {
              ctx.moveTo(canvasX, canvasY);
              started = true;
            } else {
              ctx.lineTo(canvasX, canvasY);
            }
          } else if (started) {
            // Handle boundary crossing
            const prevError = errorMin + ((x - 1) / plotWidth) * errorRange;
            const prevDeriv = derivFunc(prevError);
            const prevY = derivMidY - prevDeriv * derivScale;

            if (prevY >= derivPlotTop && prevY <= derivPlotBottom) {
              // Calculate intersection
              if (canvasY < derivPlotTop) {
                const t = (derivPlotTop - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.lineTo(intersectX, derivPlotTop);
              } else {
                const t = (derivPlotBottom - prevY) / (canvasY - prevY);
                const intersectX = padding.left + (x - 1) + t;
                ctx.lineTo(intersectX, derivPlotBottom);
              }
            }

            ctx.stroke();
            ctx.beginPath();
            started = false;
          }
        }

        if (started) {
          ctx.stroke();
        }

        // Draw discontinuity markers
        const midX = padding.left + plotWidth / 2;
        const leftDeriv = derivFunc(-0.001);
        const rightDeriv = derivFunc(0.001);

        const leftY = derivMidY - leftDeriv * derivScale;
        const rightY = derivMidY - rightDeriv * derivScale;

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
      drawDerivativeCurve((a) => a, lossColors.mse, true, 1.5);
    }

    // Draw MAE derivative if enabled
    if (showMAE) {
      drawDerivativeCurve((a) => Math.sign(a), lossColors.mae, false, 1.5);
    }

    // Draw Huber derivative (always shown)
    drawDerivativeCurve((a) => huberDerivative(a, delta), lossColors.huber, false, 2);

    // Draw Log-Cosh derivative if enabled
    if (showLogCosh) {
      drawDerivativeCurve((a) => logCoshDerivative(a), lossColors.logCosh, true, 2);
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
    const legendX = padding.left + plotWidth + 10;
    const legendY = padding.top + 20;
    const legendSpacing = 25;
    const legendItems = [];

    // Always add Huber loss to legend
    legendItems.push({
      color: lossColors.huber,
      label: `Huber Loss (δ = ${delta.toFixed(1)})`
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
    legendItems.forEach((item, index) => {
      const y = legendY + index * legendSpacing;

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
      const x = padding.left + (highlightError - errorMin) * xScale;

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
      const lossPoints = [];

      // Huber loss point
      const huberValue = huberLoss(highlightError, delta);
      const huberY = padding.top + plotHeight - Math.min(huberValue, lossMax) * lossScale;
      if (huberY >= padding.top && huberY <= padding.top + plotHeight) {
        lossPoints.push({
          color: lossColors.huber,
          y: huberY
        });
      }

      // Log-Cosh loss point
      if (showLogCosh) {
        const logCoshValue = logCoshLoss(highlightError);
        const logCoshY = padding.top + plotHeight - Math.min(logCoshValue, lossMax) * lossScale;
        if (logCoshY >= padding.top && logCoshY <= padding.top + plotHeight) {
          lossPoints.push({
            color: lossColors.logCosh,
            y: logCoshY
          });
        }
      }

      // MSE loss point
      if (showMSE) {
        const mseValue = mseLoss(highlightError);
        const mseY = padding.top + plotHeight - Math.min(mseValue, lossMax) * lossScale;
        if (mseY >= padding.top && mseY <= padding.top + plotHeight) {
          lossPoints.push({
            color: lossColors.mse,
            y: mseY
          });
        }
      }

      // MAE loss point
      if (showMAE) {
        const maeValue = maeLoss(highlightError);
        const maeY = padding.top + plotHeight - Math.min(maeValue, lossMax) * lossScale;
        if (maeY >= padding.top && maeY <= padding.top + plotHeight) {
          lossPoints.push({
            color: lossColors.mae,
            y: maeY
          });
        }
      }

      // Draw the points
      lossPoints.forEach(point => {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw points on the derivative curves
      const derivPoints = [];
      const derivMidY = derivPlotTop + plotHeight / 2;

      // Huber derivative point
      const huberDerivValue = huberDerivative(highlightError, delta);
      const huberDerivY = derivMidY - huberDerivValue * derivScale;
      if (huberDerivY >= derivPlotTop && huberDerivY <= derivPlotTop + plotHeight) {
        derivPoints.push({
          color: lossColors.huber,
          y: huberDerivY
        });
      }

      // Log-Cosh derivative point
      if (showLogCosh) {
        const logCoshDerivValue = logCoshDerivative(highlightError);
        const logCoshDerivY = derivMidY - logCoshDerivValue * derivScale;
        if (logCoshDerivY >= derivPlotTop && logCoshDerivY <= derivPlotTop + plotHeight) {
          derivPoints.push({
            color: lossColors.logCosh,
            y: logCoshDerivY
          });
        }
      }

      // MSE derivative point
      if (showMSE) {
        const mseDerivValue = highlightError;
        const mseDerivY = derivMidY - mseDerivValue * derivScale;
        if (mseDerivY >= derivPlotTop && mseDerivY <= derivPlotTop + plotHeight) {
          derivPoints.push({
            color: lossColors.mse,
            y: mseDerivY
          });
        }
      }

      // MAE derivative point
      if (showMAE) {
        const maeDerivValue = Math.sign(highlightError);
        const maeDerivY = derivMidY - maeDerivValue * derivScale;
        if (maeDerivY >= derivPlotTop && maeDerivY <= derivPlotTop + plotHeight) {
          derivPoints.push({
            color: lossColors.mae,
            y: maeDerivY
          });
        }
      }

      // Draw the derivative points
      derivPoints.forEach(point => {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw hover information
    if (hoverInfo) {
      const { x, y, error, values } = hoverInfo;

      // Background for tooltip
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.strokeStyle = '#aaa';
      ctx.lineWidth = 1;

      const tooltipWidth = 200;
      const tooltipHeight = 20 + values.length * 20;
      const tooltipX = Math.min(x + 10, width - padding.right - tooltipWidth);
      const tooltipY = Math.min(y + 10, height * 2 - 40 - tooltipHeight);

      ctx.beginPath();
      ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 5);
      ctx.fill();
      ctx.stroke();

      // Tooltip content
      ctx.fillStyle = '#333';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.font = 'bold 12px Arial';
      ctx.fillText(`Error: ${error.toFixed(2)}`, tooltipX + 10, tooltipY + 10);

      ctx.font = '12px Arial';
      values.forEach((value, index) => {
        const valueY = tooltipY + 30 + index * 20;
        ctx.fillStyle = value.color;
        ctx.fillText(`${value.label}: ${value.value.toFixed(4)}`, tooltipX + 10, valueY);
      });
    }
  }, [delta, showLogCosh, showMSE, showMAE, highlightError, hoverInfo]);

  // Handle mouse movement over canvas for hover effects
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = { left: 60, right: 60, top: 50, bottom: 40 };
    const height = canvas.height / 2;
    const plotWidth = canvas.width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Check if mouse is in the plot area (either top or bottom)
    const isInTopPlot = (
      x >= padding.left && x <= padding.left + plotWidth &&
      y >= padding.top && y <= padding.top + plotHeight
    );

    const derivPlotTop = height + padding.top;
    const isInBottomPlot = (
      x >= padding.left && x <= padding.left + plotWidth &&
      y >= derivPlotTop && y <= derivPlotTop + plotHeight
    );

    if (isInTopPlot || isInBottomPlot) {
      // Calculate corresponding error value
      const errorMin = -5;
      const errorMax = 5;
      const errorRange = errorMax - errorMin;

      const error = errorMin + ((x - padding.left) / plotWidth) * errorRange;

      // Set highlight position
      setHighlightError(error);

      // Calculate loss and derivative values
      const values = [];

      // Huber loss
      const huberValue = huberLoss(error, delta);
      values.push({
        label: `Huber Loss (δ = ${delta.toFixed(1)})`,
        value: huberValue,
        color: '#4682B4'
      });

      const huberDerivValue = huberDerivative(error, delta);
      values.push({
        label: 'Huber Derivative',
        value: huberDerivValue,
        color: '#4682B4'
      });

      // Log-Cosh loss
      if (showLogCosh) {
        const logCoshValue = logCoshLoss(error);
        values.push({
          label: 'Log-Cosh Loss',
          value: logCoshValue,
          color: '#9C27B0'
        });

        const logCoshDerivValue = logCoshDerivative(error);
        values.push({
          label: 'Log-Cosh Derivative',
          value: logCoshDerivValue,
          color: '#9C27B0'
        });
      }

      // MSE loss
      if (showMSE) {
        const mseValue = mseLoss(error);
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
        const maeValue = maeLoss(error);
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

      setHoverInfo({ x, y, error, values });
    } else {
      setHighlightError(null);
      setHoverInfo(null);
    }
  };

  const handleMouseLeave = () => {
    setHighlightError(null);
    setHoverInfo(null);
  };

  return (
      <div className="p-4 w-full flex flex-col items-center">
        <h2 className="text-xl font-bold mb-4">MAE vs MSE vs Huber vs Log-Cosh Loss Comparison</h2>

        <div className="w-full max-w-4xl mb-4 flex items-center gap-4 justify-center">
          {/* Range input group */}
          <div className="flex items-center space-x-2">
            <label htmlFor="delta" className="font-medium">Huber δ:</label>
            <input
                id="delta"
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={delta}
                onChange={(e) => setDelta(parseFloat(e.target.value))}
                className="w-40"
            />
            <span className="w-10 text-center">{delta.toFixed(1)}</span>
          </div>

          {/* Checkboxes group */}
          <div className="flex items-center space-x-2">
            <label className="inline-flex items-center">
              <input
                  type="checkbox"
                  checked={showLogCosh}
                  onChange={(e) => setShowLogCosh(e.target.checked)}
                  className="mr-1"
              />
              Log-Cosh
            </label>

            <label className="inline-flex items-center">
              <input
                  type="checkbox"
                  checked={showMSE}
                  onChange={(e) => setShowMSE(e.target.checked)}
                  className="mr-1"
              />
              MSE
            </label>

            <label className="inline-flex items-center">
              <input
                  type="checkbox"
                  checked={showMAE}
                  onChange={(e) => setShowMAE(e.target.checked)}
                  className="mr-1"
              />
              MAE
            </label>
          </div>
        </div>

        <canvas
            ref={canvasRef}
            width={800}
            height={800}
            className="border border-gray-300 bg-white"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
        />
      </div>
  );
};