import numpy as np
import plotly.graph_objects as go


def make_flood_animation(
    prediction,
    x_grid,
    y_grid,
    title="JalRaksha 3D Flood Digital Twin",
):

    depth = np.asarray(
        prediction,
        dtype=float
    )

    x_grid = np.asarray(
        x_grid,
        dtype=float
    )

    y_grid = np.asarray(
        y_grid,
        dtype=float
    )

    X, Y = np.meshgrid(
        x_grid,
        y_grid
    )

    depth = np.maximum(
        depth,
        0.0
    )

    maximum_depth = float(
        depth.max()
    )

    frames = []

    n_frames = 12

    for i in range(n_frames):

        progress = (
            i / (n_frames - 1)
        )

        # Visual flood-spread ramp.
        #
        # IMPORTANT:
        # This is a visualization of the final predicted
        # depth field. It is NOT a transient hydrodynamic
        # forecast.

        threshold = (
            maximum_depth
            * (1.0 - progress)
        )

        visible = depth >= threshold

        z = np.where(
            visible,
            depth * progress,
            np.nan,
        )

        frame = go.Frame(
            name=f"frame_{i}",
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=z,
                    surfacecolor=depth,
                    cmin=0,
                    cmax=max(
                        maximum_depth,
                        1e-6
                    ),
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(
                        title="Depth (m)"
                    ),
                )
            ],
        )

        frames.append(frame)

    # Initial frame
    first_progress = 0.0

    threshold = maximum_depth

    initial_visible = depth >= threshold

    initial_z = np.where(
        initial_visible,
        depth * first_progress,
        np.nan,
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=initial_z,
                surfacecolor=depth,
                cmin=0,
                cmax=max(
                    maximum_depth,
                    1e-6
                ),
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title="Depth (m)"
                ),
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        title=title,
        height=620,
        margin=dict(
            l=0,
            r=0,
            t=55,
            b=0,
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Flood depth (m)",
            aspectmode="manual",
            aspectratio=dict(
                x=6,
                y=1.2,
                z=0.6,
            ),
            camera=dict(
                eye=dict(
                    x=1.5,
                    y=1.2,
                    z=0.8,
                )
            ),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Animate Flood Spread",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": 350,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 150,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.12,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                len=0.82,
                currentvalue=dict(
                    prefix="Flood spread: "
                ),
                steps=[
                    dict(
                        label=f"{int(i/(n_frames-1)*100)}%",
                        method="animate",
                        args=[
                            [f"frame_{i}"],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": True,
                                },
                            },
                        ],
                    )
                    for i in range(n_frames)
                ],
            )
        ],
    )

    return fig


if __name__ == "__main__":

    print(
        "3D flood renderer loaded successfully."
    )
