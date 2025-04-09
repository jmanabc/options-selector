window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        capture_click: function(clickData) {
            if (!clickData || !clickData.points || clickData.points.length === 0) {
                return null;
            }
            // Return the x value of the first clicked bar (the bin midpoint)
            return clickData.points[0].x;
        }
    }
});
