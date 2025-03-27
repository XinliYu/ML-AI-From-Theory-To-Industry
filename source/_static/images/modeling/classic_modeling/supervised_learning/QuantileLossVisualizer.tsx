import React, { useState, useEffect, useRef } from 'react';

const QuantileLossVisualizer = () => {
  // Canvas and visualization parameters
  const canvasRef = useRef(null);
  const [tau, setTau] = useState(0.5);
  const [highlightError, setHighlightError] = useState(0);
  const [hoverInfo, setHoverInfo] = useState(null);

  // Calculate quantile loss for a given error and tau
  const quantileLoss = (error, quantile) => {
    return error >= 0
      ? quantile * error
      : (1 - quantile) * (-error);
  };

  // Calculate derivative of quantile loss
  const quantileLossDerivative = (error, quantile) => {
    // Note: The derivative is not defined at error = 0, but we approximate
    if (Math.abs(error) < 0.001) return 0;
    return error > 0 ? -quantile : (1 - quantile);
  };

  // Draw the visualization on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    // Define visualization parameters
    const padding = { left: 60, right: 100, top: 60, bottom: 40 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height / 2 - padding.top - padding.bottom / 2;

    // Error range from -5 to 5
    const errorMin = -5;
    const errorMax = 5;
    const errorRange = errorMax - errorMin;

    // X-axis scale (mapping error values to canvas coordinates)
    const xScale = plotWidth / errorRange;

    // Y-axis scales for loss and derivative
    const lossMax = 5;
    const lossScale = plotHeight / lossMax;

    const derivMin = -1;
    const derivMax = 1;
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

    // Draw different quantile loss functions
    const lossColors = {
      median: '#00C853', // Median (τ = 0.5)
      custom: '#FF6D00', // Custom τ value
    };

    // Function to draw a loss curve
    const drawLossCurve = (quantileValue, color, label, yOffset, lineWidth = 2) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      for (let x = 0; x <= plotWidth; x += 1) {
        const error = errorMin + (x / plotWidth) * errorRange;
        const loss = quantileLoss(error, quantileValue);

        // Ensure loss is within visible range
        const clampedLoss = Math.min(loss, lossMax);

        const canvasX = padding.left + x;
        const canvasY = padding.top + plotHeight - clampedLoss * lossScale;

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
      drawLossCurve(tau, lossColors.custom, `Quantile Regression`, 40);
    }

    // Draw the derivative plot (bottom)
    // --------------------------------
    const derivPlotTop = padding.top * 2 + plotHeight;

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Y-axis
    ctx.moveTo(padding.left, derivPlotTop);
    ctx.lineTo(padding.left, derivPlotTop + plotHeight);

    // X-axis - centered at the middle of the plot height for derivative
    const xAxisY = derivPlotTop + plotHeight / 2;
    ctx.moveTo(padding.left, xAxisY);
    ctx.lineTo(padding.left + plotWidth, xAxisY);

    ctx.stroke();

    // Draw grid lines and labels for derivative plot
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;

    // Y-axis grid lines and labels
    const derivTickValues = [-1, -0.5, 0, 0.5, 1];

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';

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

    // X-axis grid lines reuse the same values as the top plot
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

    // Function to draw a derivative curve
    const drawDerivativeCurve = (quantileValue, color, discontinuities = true) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;

      if (discontinuities) {
        // Draw with visible discontinuity at x = 0
        // Negative side
        ctx.beginPath();
        for (let x = 0; x < plotWidth / 2; x += 1) {
          const error = errorMin + (x / plotWidth) * errorRange;
          const deriv = quantileLossDerivative(error, quantileValue);

          const canvasX = padding.left + x;
          const canvasY = derivPlotTop + plotHeight / 2 - deriv * derivScale;

          if (x === 0) {
            ctx.moveTo(canvasX, canvasY);
          } else {
            ctx.lineTo(canvasX, canvasY);
          }
        }
        ctx.stroke();

        // Positive side
        ctx.beginPath();
        for (let x = plotWidth / 2 + 1; x <= plotWidth; x += 1) {
          const error = errorMin + (x / plotWidth) * errorRange;
          const deriv = quantileLossDerivative(error, quantileValue);

          const canvasX = padding.left + x;
          const canvasY = derivPlotTop + plotHeight / 2 - deriv * derivScale;

          if (x === plotWidth / 2 + 1) {
            ctx.moveTo(canvasX, canvasY);
          } else {
            ctx.lineTo(canvasX, canvasY);
          }
        }
        ctx.stroke();

        // Draw points at the discontinuity
        const midX = padding.left + plotWidth / 2;

        // Point at (0, -(1-τ))
        const y1 = derivPlotTop + plotHeight / 2 - (-(1 - quantileValue)) * derivScale;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(midX, y1, 4, 0, Math.PI * 2);
        ctx.fill();

        // Point at (0, -τ)
        const y2 = derivPlotTop + plotHeight / 2 - (-quantileValue) * derivScale;
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
      const x = padding.left + (highlightError - errorMin) * xScale;

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
      const highlightPoints = [];

      const medianLoss = quantileLoss(highlightError, 0.5);
      const y = padding.top + plotHeight - Math.min(medianLoss, lossMax) * lossScale;
      highlightPoints.push({ color: lossColors.median, y });

      if (tau !== 0.5) {
        const customLoss = quantileLoss(highlightError, tau);
        const y = padding.top + plotHeight - Math.min(customLoss, lossMax) * lossScale;
        highlightPoints.push({ color: lossColors.custom, y });
      }

      // Draw the points
      highlightPoints.forEach(point => {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw derivative points
      const derivPoints = [];

      const medianDeriv = quantileLossDerivative(highlightError, 0.5);
      const yDeriv = derivPlotTop + plotHeight / 2 - medianDeriv * derivScale;
      derivPoints.push({ color: lossColors.median, y: yDeriv });

      if (tau !== 0.5) {
        const customDeriv = quantileLossDerivative(highlightError, tau);
        const yDeriv = derivPlotTop + plotHeight / 2 - customDeriv * derivScale;
        derivPoints.push({ color: lossColors.custom, y: yDeriv });
      }

      // Draw the derivative points
      derivPoints.forEach(point => {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, point.y, 5, 0, Math.PI * 2);
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
      const tooltipY = Math.min(y + 10, height - padding.bottom - tooltipHeight);

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

    // Draw legend
    const legendX = padding.left + plotWidth - 120;
    const legendY = padding.top + 30;


  }, [tau, highlightError, hoverInfo]);

  // Handle mouse movement over canvas for hover effects
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = { left: 60, right: 100, top: 60, bottom: 40 };
    const plotWidth = canvas.width - padding.left - padding.right;
    const plotHeight = canvas.height / 2 - padding.top - padding.bottom / 2;

    // Check if mouse is in the plot area (either top or bottom)
    const isInTopPlot = (
      x >= padding.left && x <= padding.left + plotWidth &&
      y >= padding.top && y <= padding.top + plotHeight
    );

    const derivPlotTop = padding.top * 2 + plotHeight;
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

      // Median loss and derivative
      const medianLoss = quantileLoss(error, 0.5);
      values.push({
        label: 'Median Loss (τ = 0.5)',
        value: medianLoss,
        color: '#00C853'
      });

      const medianDeriv = quantileLossDerivative(error, 0.5);
      values.push({
        label: 'Median Derivative',
        value: medianDeriv,
        color: '#00C853'
      });

      // Custom quantile loss and derivative (if different from median)
      if (tau !== 0.5) {
        const customLoss = quantileLoss(error, tau);
        values.push({
          label: `Quantile Loss (τ = ${tau.toFixed(2)})`,
          value: customLoss,
          color: '#FF6D00'
        });

        const customDeriv = quantileLossDerivative(error, tau);
        values.push({
          label: `Quantile Derivative (τ = ${tau.toFixed(2)})`,
          value: customDeriv,
          color: '#FF6D00'
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
      <div className="mb-4 w-full max-w-4xl">
        <p className="mb-2">
          Quantile regression estimates a specific quantile of the conditional distribution rather than the mean. The parameter τ ∈ (0,1) determines
          which quantile is targeted (e.g., τ = 0.5 for median, τ = 0.9 for 90th percentile).
        </p>
        <p className="mb-2">
          <strong>The loss function has an asymmetric weighting based on τ:</strong>
        </p>
        <ul className="list-disc ml-8 mb-2">
          <li>When prediction is too high (error {'<'} 0): weighted by (1-τ)</li>
          <li>When prediction is too low (error {'>'} 0): weighted by τ</li>
        </ul>
      </div>

      <div className="w-full max-w-4xl mb-4 flex items-center">
        <label htmlFor="tau" className="font-medium mr-2">Quantile (τ):</label>
        <input
          id="tau"
          type="range"
          min="0.01"
          max="0.99"
          step="0.01"
          value={tau}
          onChange={(e) => setTau(parseFloat(e.target.value))}
          className="w-60 mr-2"
        />
        <span className="w-16 text-center font-mono">{tau.toFixed(2)}</span>
      </div>

      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="border border-gray-300 bg-white"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />

      <div className="mt-6 w-full max-w-4xl text-sm">
        <h3 className="font-bold text-lg mb-2">Key Properties of Quantile Loss:</h3>
        <ul className="list-disc ml-6 space-y-2">
          <li>
            <strong>Asymmetric Weighting:</strong> The loss function penalizes overestimation and underestimation differently
            based on the value of τ.
          </li>
          <li>
            <strong>Discontinuous Derivative:</strong> The derivative has a discontinuity at zero, which is a key feature that
            makes the loss function pull the predictions toward the desired quantile.
          </li>
          <li>
            <strong>τ = 0.5 (Median):</strong> This special case gives equal weight to positive and negative errors,
            resulting in median regression (equivalent to minimizing Mean Absolute Error).
          </li>
          <li>
            <strong>Risk-Averse Models:</strong> High values of τ (e.g., 0.9) are especially useful for risk-averse
            predictions, where overestimation is more costly than underestimation.
          </li>
        </ul>
      </div>
    </div>
  );
};

export default QuantileLossVisualizer;