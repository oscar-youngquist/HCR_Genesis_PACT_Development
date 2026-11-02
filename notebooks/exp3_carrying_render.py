"""Standalone carrying figure: only NumPy, pandas, Matplotlib and packaged data.

This file is copied unchanged to render_figure.py in the refinement package.
No repository, robot model, policy, or simulator is needed to re-bin or render.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ('front_change_n', 'rear_change_n', 'load_n', 'arm_ff_change_nm',
           'leg_ff_change_nm', 'ee_error_m', 'geometry_error_m')
DEFAULT_CONFIG = dict(
    size_inches=[7.0, 2.2], bin_width_seconds=.25, smoothing_bins=1,
    baseline_interval_seconds=None, baseline_min_coverage=1.0,
    font_family='Times New Roman', fallback_font_family='DejaVu Serif',
    title_font_size=8.5, axis_label_font_size=8, tick_font_size=7,
    legend_font_size=7, annotation_font_size=6.5,
    colors={'PILOT': '#0072B2', 'A6': '#D55E00'}, band_alpha=.14,
    line_alpha=1., line_width=1.05, secondary_linestyle='--',
    load_color='#555555', load_alpha=.75, event_alpha=.3,
    method_order=['PILOT', 'A6'], x_limits=None,
    y_limits=[None, None, None],
    titles=['A  Measured Support\nRedistribution', 'B  Arm and Leg\nCompensation',
            'C  Commanded Geometry\nand EE Tracking'],
    y_labels=[r'$\Delta$ Normal Support [N]', r'$\Delta$ Torque RMS [N$\cdot$m]', 'EE Error [m]'],
    quantity_labels=[['Front', 'Rear', 'Downward EE Load'], ['Arm', 'Leg'], ['Actual', 'Commanded Geometry']],
    x_label='Time [s]', panel_legend_locations=['upper left', 'upper left', 'upper right'],
    method_legend_location='lower center', method_legend_anchor=[.53, -.005],
    layout_rect=[0, .10, 1, 1], layout_pad=.35, layout_w_pad=.7,
    show_event_labels=True, png_dpi=300,
)


def edges(metadata, config):
    width = float(config['bin_width_seconds'])
    if not np.isfinite(width) or width <= 0:
        raise ValueError('bin_width_seconds must be positive')
    # Insert event/ramp boundaries into the common grid: no bin spans stages.
    boundaries = np.asarray(metadata['stage_boundaries_seconds'], float)
    return np.unique(np.round(np.r_[np.arange(0, boundaries[-1], width), boundaries], 10))


def _segments(d, valid, stage):
    continuous = ((np.diff(d['step']) == 1) & valid[:-1] & valid[1:]
                  & (stage[:-1] == stage[1:])
                  & np.isclose(d['time_seconds'][1:], d['successor_time_seconds'][:-1], atol=1e-7, rtol=0))
    return np.cumsum(np.r_[True, ~continuous])


def baseline(d, valid, interval, minimum):
    """Baseline RMS pools joint-substep squares, not per-step or per-bin RMS."""
    lo, hi = interval
    select = valid & (d['time_seconds'] >= lo-1e-8) & (d['successor_time_seconds'] <= hi+1e-8)
    ix = np.flatnonzero(select)
    duration = np.sum((d['successor_time_seconds']-d['time_seconds'])[select])
    continuous = (len(ix) > 0 and np.all(np.diff(d['step'][ix]) == 1)
                  and np.allclose(d['time_seconds'][ix[1:]], d['successor_time_seconds'][ix[:-1]], atol=1e-7, rtol=0))
    usable = continuous and duration >= (hi-lo)*minimum-1e-7
    result = dict(baseline_valid=bool(usable), baseline_duration_s=float(duration),
                  baseline_samples=int(select.sum()))
    for region in ('front', 'rear'):
        value = d[region+'_normal_n'][select]
        result[region+'_baseline_n'] = float(value.mean(dtype=np.float64)) if usable and np.isfinite(value).all() else np.nan
    for region in ('arm', 'leg'):
        count = int(d[region+'_ff_count'][select].sum())
        square = float(d[region+'_ff_sumsq'][select].sum())
        result[region+'_baseline_nm'] = np.sqrt(square/count) if usable and count else np.nan
        result[region+'_baseline_joint_substeps'] = count
        result[region+'_baseline_sumsq'] = square
    return result


def reduce_trial(d, metadata, config, identity):
    order = np.argsort(d['step']); d = {k: v[order] for k, v in d.items()}
    if np.any(np.diff(d['step']) <= 0):
        raise ValueError('Duplicate/out-of-order trial steps')
    terminal = d['physical_failure'] | d['artificial_reset']
    # Later rows cannot resurrect a failed/reset trial, even if their flags are bad.
    valid = d['valid'] & d['alive_for_metrics'] & ~np.maximum.accumulate(terminal)
    baseline_interval = config.get('baseline_interval_seconds') or metadata['baseline_interval_seconds']
    lo, hi = baseline_interval
    allowed = metadata['baseline_interval_seconds']
    if not (allowed[0] <= lo < hi <= allowed[1]):
        raise ValueError('Baseline must lie within metadata-verified unloaded locomotion')
    minimum = float(config['baseline_min_coverage'])
    if not 0 < minimum <= 1:
        raise ValueError('baseline_min_coverage must be in (0,1]')
    base = baseline(d, valid, baseline_interval, minimum)
    grid = edges(metadata, config); rows = []; dropped = 0
    width = config['smoothing_bins']
    if isinstance(width, bool) or not isinstance(width, int) or width < 1 or width % 2 != 1:
        raise ValueError('smoothing_bins must be a positive odd integer')
    failure_time = d['successor_time_seconds'][np.flatnonzero(terminal)[0]] if terminal.any() else np.inf
    for metric in METRICS:
        successor = metric == 'ee_error_m'
        stage = d['stage_successor' if successor else 'stage_interval']
        # Load/torque are interval averages at interval midpoint. Geometry uses
        # base/reference at t; actual EE tracking uses the physical successor t+dt.
        timestamp = (d['successor_time_seconds'] if successor else d['time_seconds'] if metric=='geometry_error_m'
                     else .5*(d['time_seconds']+d['successor_time_seconds']))
        bins = np.searchsorted(grid, np.round(timestamp,10), side='right')-1
        bins = np.minimum(bins, len(grid)-2)  # Include the horizon endpoint in the final bin.
        region = metric.split('_')[0]
        if '_ff_' in metric:
            total, counts = d[region+'_ff_sumsq'], d[region+'_ff_count']
            offset = base[region+'_baseline_nm']
        else:
            value = d[region+'_normal_n'] if region in ('front','rear') else d[metric]
            value = np.asarray(value,dtype=np.float64)  # Match existing Exp3 grouped reduction precision.
            total = np.where(np.isfinite(value), value, 0.)
            counts = np.isfinite(value).astype(int)
            offset = base[region+'_baseline_n'] if region in ('front','rear') else 0.
        segment = _segments(d, valid & (counts > 0), stage)
        mask = valid & (counts > 0) & (bins >= 0) & np.isfinite(offset)
        part = pd.DataFrame(dict(stage=stage[mask], bin=bins[mask], segment=segment[mask],
                                 total=total[mask], denominator=counts[mask], samples=1,
                                 first=timestamp[mask], last=timestamp[mask]))
        if part.empty: continue
        grouped = part.groupby(['stage','bin'], sort=True)
        stats = grouped.agg(total=('total','sum'), denominator=('denominator','sum'), samples=('samples','sum'),
                            first=('first','min'), last=('last','max'), segment=('segment','first'), segments=('segment','nunique')).reset_index()
        dropped += int((stats.segments != 1).sum())
        stats = stats[stats.segments == 1].copy()  # Never pool disconnected pieces of one bin.
        new_segment = (stats.stage.diff().ne(0) | stats.segment.diff().ne(0) | stats.bin.diff().ne(1)).cumsum()
        if width > 1:
            # Optional smoothing re-pools native sufficient statistics BEFORE
            # taking square roots, and never spans a missing bin or stage boundary.
            for _, indices in stats.groupby(new_segment).groups.items():
                cols = ['total','denominator','samples']
                stats.loc[indices, cols] = stats.loc[indices, cols].rolling(width,center=True,min_periods=1).sum().to_numpy()
        values = stats.total / stats.denominator
        if '_ff_' in metric: values = np.sqrt(values)
        stats['value'] = values-offset
        for r in stats.itertuples(index=False):
            end = grid[r.bin+1]
            rows.append(dict(**identity,metric=metric,stage=int(r.stage),bin=int(r.bin),
                bin_start_s=grid[r.bin],bin_end_s=end,time_seconds=(grid[r.bin]+end)/2,
                value=r.value,baseline=offset,samples=int(r.samples),denominator=int(r.denominator),
                first_sample_s=r.first,last_sample_s=r.last,
                surviving_at_bin_end=bool(failure_time > end+1e-8 and d['successor_time_seconds'][-1]>=end-1e-8)))
    return rows, dict(**identity,**base), dropped


def aggregate(trials):
    keys = ['method','checkpoint','metric','stage','bin','bin_start_s','bin_end_s','time_seconds']
    if trials.empty: raise ValueError('No usable plotted values')
    seeds = trials.groupby(keys+['evaluation_seed'],dropna=False).agg(
        mean=('value','mean'), contributing_trials=('trial','size'),
        surviving_trials=('surviving_at_bin_end','sum'), valid_samples=('samples','sum'),
        denominator=('denominator','sum')).reset_index()
    summary = seeds.groupby(keys,dropna=False).agg(mean=('mean','mean'),sd=('mean','std'),
        contributing_seeds=('evaluation_seed','size'),contributing_trials=('contributing_trials','sum'),
        surviving_trials=('surviving_trials','sum'),valid_samples=('valid_samples','sum'),
        denominator=('denominator','sum')).reset_index()
    return seeds, summary


def calculate(directory, config):
    directory = Path(directory); meta = json.loads((directory/'metadata.json').read_text())
    rows, bases, drops = [], [], 0
    for source in meta['sources']:
        with np.load(directory/source['data_file'],allow_pickle=False) as f:
            data = {k:f[k] for k in f.files}
        for tid in np.unique(data['trial']):
            selected = data['trial'] == tid
            d = {k:v[selected] for k,v in data.items()}
            identity = dict(method=source['method'],checkpoint=source['checkpoint_sha256'],
                            evaluation_seed=source['evaluation_seed'],trial=int(tid))
            r,b,n = reduce_trial(d,meta,config,identity); rows.extend(r); bases.append(b); drops+=n
    trials = pd.DataFrame(rows); seeds, summary = aggregate(trials)
    return trials, pd.DataFrame(bases), seeds, summary, drops


def draw(summary, metadata, config, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    requested = config['font_family']
    try: font_manager.findfont(requested, fallback_to_default=False); font=requested
    except ValueError: font=config['fallback_font_family']
    settings = {'font.family':font,'font.size':config['tick_font_size'],'pdf.fonttype':42,
                'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix'}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1,3,figsize=config['size_inches'])
        quantities = [('front_change_n','rear_change_n'),('arm_ff_change_nm','leg_ff_change_nm'),('ee_error_m','geometry_error_m')]
        grid=edges(metadata,config)
        for panel,(ax,metrics) in enumerate(zip(axes,quantities)):
            for method in config['method_order']:
                for j,metric in enumerate(metrics):
                    frame=summary[(summary.method==method)&(summary.metric==metric)]
                    # Separate stages and insert NaNs for missing bins: no interpolation.
                    for stage,g in frame.groupby('stage'):
                        bins=np.arange(len(grid)-1)
                        in_stage=(grid[:-1]>=metadata['stage_boundaries_seconds'][stage]-1e-8)&(grid[1:]<=metadata['stage_boundaries_seconds'][stage+1]+1e-8)
                        p=g.set_index('bin').reindex(bins[in_stage]); x=(grid[:-1]+grid[1:])[in_stage]/2
                        color=config['colors'][method]
                        ax.plot(x,p['mean'],color=color,ls='-' if j==0 else config['secondary_linestyle'],
                                lw=config['line_width'],alpha=config['line_alpha'])
                        ax.fill_between(x,p['mean']-p.sd,p['mean']+p.sd,color=color,alpha=config['band_alpha'],linewidth=0)
            labels=config['quantity_labels'][panel]
            handles=[Line2D([],[],color='black',ls=ls,lw=1,label=label) for ls,label in zip(('-',config['secondary_linestyle']),labels)]
            if panel==0:
                # The load is a reference in N, not an additional support estimate.
                load=summary[summary.metric=='load_n'].groupby('bin')['mean'].mean()
                ax.stairs(load.reindex(np.arange(len(grid)-1)).to_numpy(),grid,color=config['load_color'],
                          ls='-.',lw=.85,alpha=config['load_alpha'],baseline=None)
                handles.append(Line2D([],[],color=config['load_color'],ls='-.',lw=.85,label=labels[2]))
            for event in metadata['events']:
                ax.axvline(event['time_seconds'],color='black',lw=.5,ls=':',alpha=config['event_alpha'])
            if config['show_event_labels']:
                event=next(e for e in metadata['events'] if e['kind']==('stop' if panel==1 else 'release')) if panel else None
                if event:
                    ax.text(event['time_seconds'],1.01,event['kind'].title(),transform=ax.get_xaxis_transform(),
                            ha='center',va='bottom',fontsize=config['annotation_font_size'])
            ax.axhline(0,color='#aaaaaa',lw=.4,zorder=0)
            ax.set_title(config['titles'][panel],loc='left',fontsize=config['title_font_size'],pad=5)
            ax.set_ylabel(config['y_labels'][panel],fontsize=config['axis_label_font_size'],labelpad=2)
            ax.set_xlabel(config['x_label'],fontsize=config['axis_label_font_size'],labelpad=1)
            ax.set_xlim(config['x_limits'] or [metadata['baseline_interval_seconds'][0],grid[-1]])
            if config['y_limits'][panel] is not None: ax.set_ylim(config['y_limits'][panel])
            ax.tick_params(labelsize=config['tick_font_size'],length=2,pad=2)
            ax.spines[['top','right']].set_visible(False)
            ax.legend(handles=handles,loc=config['panel_legend_locations'][panel],fontsize=config['legend_font_size'],
                      framealpha=.85,edgecolor='none',borderpad=.15,labelspacing=.2,handlelength=1.7,handletextpad=.4)
        handles=[Line2D([],[],color=config['colors'][m],lw=1.4,label=m) for m in config['method_order']]
        fig.legend(handles=handles,loc=config['method_legend_location'],bbox_to_anchor=config['method_legend_anchor'],
                   ncol=len(handles),frameon=False,fontsize=config['legend_font_size'],borderaxespad=0)
        fig.tight_layout(rect=config['layout_rect'],pad=config['layout_pad'],w_pad=config['layout_w_pad'])
        for extension in ('pdf','svg','png'):
            fig.savefig(Path(output)/('exp3_carrying_compact.'+extension),dpi=config['png_dpi'])
        plt.close(fig)
    return font


def render(directory, config_path=None):
    directory=Path(directory)
    config={**DEFAULT_CONFIG,**json.loads(Path(config_path or directory/'plot_config.json').read_text())}
    metadata=json.loads((directory/'metadata.json').read_text())
    trials,baselines,seeds,summary,drops=calculate(directory,config)
    trials.to_csv(directory/'plotted_trials.csv.gz',index=False,compression='gzip')
    baselines.to_csv(directory/'baselines.csv',index=False)
    seeds.to_csv(directory/'plotted_seeds.csv',index=False)
    summary.to_csv(directory/'plotted_summary.csv',index=False)
    font=draw(summary,metadata,config,directory)
    report=dict(font_used=font,font_fallback=font!=config['font_family'],gap_split_metric_bins_excluded=drops,
                invalid_trial_baselines=int((~baselines.baseline_valid).sum()),
                aggregation='Trials first, equal evaluation seeds, sample SD (ddof=1); fixed checkpoint per method',
                config=config)
    (directory/'render_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args(); print(json.dumps(render(args.package,args.config),indent=2))
