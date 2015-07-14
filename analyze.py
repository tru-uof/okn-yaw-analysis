# encoding: utf-8

import sys

import numpy as np
import tables
from constants import *
from memorize import memorize
from utils import get_cleaned_data, get_merged_data, get_interp_naksu, px2heading, get_pursuit_fits, get_angled_range_data2, get_range_data, get_pooled, get_condition_laps, filter_by_treatment
from tru.rec import foreach, rowstack, colstack, rec_degroup, groupby_multiple, groupby, groupby_i, append_field
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import itertools

from matplotlib.backends.backend_pdf import PdfPages

from tru.se_scenecam import screencapture_to_angles

def toyota_yaw_rate_in_degrees(rate):
    return -0.25*rate + 128

def plot_curve_map():
    import pymapdl
    ci = 3
    data = get_range_data(APPROACH[ci][0], CORNERING[ci][1])

    d = foreach(data, ['session_id', 'lap'])
    d = d[np.random.randint(0, len(d))]

    ad = d[d['dist'] < ENTRY[ci][0]]
    ac = ad['latitude'], ad['longitude']

    ed = d[(d['dist'] > ENTRY[ci][0]) & 
           (d['dist'] < CORNERING[ci][0])]
    ec = ed['latitude'], ed['longitude']
    
    tpd = d[(d['dist'] > TURNIN[ci][0]) & 
           (d['dist'] < TURNIN[ci][1])]
    tpstart = tpd[ tpd['dist'] == np.min(tpd['dist']) ]
    tpstart = tpstart['latitude'], tpstart['longitude']
    tpstop = tpd[ tpd['dist'] == np.max(tpd['dist']) ]
    tpstop = tpstop['latitude'], tpstop['longitude']
    #print tpstart, tpstop

    cd = d[d['dist'] > CORNERING[ci][0]]
    cc = cd['latitude'], cd['longitude']
    cstart = cd[ cd['dist'] == np.min(cd['dist']) ]
    cstop = cd[ cd['dist'] == np.max(cd['dist']) ]
    #print cstart['latitude'], cstart['longitude']
    #print cstop['latitude'], cstop['longitude']
    
    c_end = np.max(d['dist']) - 35
    c_end = d['dist'].searchsorted(c_end)
    c_end = d[c_end]
    c_end = c_end['latitude'], c_end['longitude']
    #print c_end

    extent = ((np.min(d['latitude']), np.max(d['latitude'])),
        (np.min(d['longitude']), np.max(d['longitude'])))
    
    MAP = "http://tiles.kartat.kapsi.fi/ortokuva/%(z)s/%(x)s/%(y)s.jpg"
    #MAP = pymapdl.OSM_MAPNIK
    tile, map_extent = pymapdl.fetch_containing_map(extent[0], extent[1], 18, MAP)

    xs, ys = np.subtract(*map_extent[0]), np.subtract(*map_extent[1])
    map_extent = list(itertools.chain(*map_extent))

    tile = np.array(tile)
    #tile = np.rot90(tile)
    aspect = np.abs(tile.shape[1]/float(tile.shape[0])*(xs/float(ys)))
    #print aspect
    plt.imshow(tile, extent=map_extent, aspect=aspect)

    plt.ylabel('Latitude (degrees)')
    plt.xlabel('Longitude (degrees)')

    #plt.plot(*ac[::-1], color='blue', linestyle="dotted", linewidth=3, alpha=0.8)
    plt.plot(*ec[::-1], color='white', linestyle="dotted", linewidth=3, alpha=0.8)
    plt.plot(*cc[::-1], color='white', linewidth=3, alpha=0.8)
    #plt.plot(*c_end[::-1], color='red', marker='o', alpha=1)
    #plt.plot(*tpstart[::-1], color='blue', marker='o', alpha=1)
    #plt.plot(*tpstop[::-1], color='blue', marker='o', alpha=1)

    axis = map_extent
    #axis[2], axis[3] = axis[3], axis[2]
    #axis[0], axis[1] = axis[1], axis[0]
    plt.axis(axis)
    plt.show()


def t_ci(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    w = scipy.stats.t.ppf(1.0-alpha/2.0, n-1)*(std/np.sqrt(n))
    return mean-w, mean+w


def plot_aoi_catch():
    ci = 1
    aois = np.arange(1, 11, 1)
        
    def get_aoi_catch(data):
        res = []
        for k, d in zip(*foreach(data, ['session_id'], return_keys=True)):
            sid = k['session_id']
            dx = px2heading(d['scenecam_x']) - px2heading(d['naksu_x'])
            dy = py2pitch(d['scenecam_y']) - py2pitch(d['naksu_y'])
            delta = np.sqrt(dx**2 + dy**2)
            delta = delta[np.isfinite(delta)]
    
            for aoi in aois:
                catch = np.sum(delta < aoi)/float(len(delta))
                res.append((sid, aoi, catch))
    
        res = np.rec.fromrecords(res, names='session_id,aoi,catch')
        return res
    
    
    data = get_range_data(CORNERING[ci][0], CORNERING[ci][1])
    data = data[data['g_direction_q'] > 0.4]
    
    #plt.subplot(1,2,1)
    #plt.title('Cornering')
    res = get_aoi_catch(data)

    plt.plot(res['aoi']+0.1, res['catch']*100, 'ro', alpha=0.8)

    med = np.array([np.median(r['catch']) for r in foreach(res, ['aoi'])])
    plt.plot(aois, med*100, color='red')
    
        
    plt.ylim(0, 100)
    plt.ylabel('Share of gazes in TP AOI (%)')
    plt.xlabel('AOI size (degrees)')
    
    data = get_range_data(ENTRY[ci][0], ENTRY[ci][1])
    data = data[data['g_direction_q'] > 0.4]
    res = get_aoi_catch(data)

    #plt.subplot(1,2,2)
    #plt.title('Entry')
    med = np.array([np.median(r['catch']) for r in foreach(res, ['aoi'])])
    plt.plot(aois, med*100, color='green', linestyle='dashed')
    
    plt.plot(res['aoi']-0.1, res['catch']*100, 'gx', alpha=0.8)
    med = np.array([np.median(r['catch']) for r in foreach(res, ['aoi'])])


    plt.ylim(0, 100)
    #plt.ylabel('Share of gazes in TP AOI (%)')
    plt.xlabel('AOI size (degrees)')
    plt.show()

def pursuit_endpoint_distributions():
    from segreg import piecewise_linear_regression, \
        piecewise_linear_regression_pseudo_em,\
        regression_interpolator, slope_density, slope_stats
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.stats import gaussian_kde

    ci = 1
    data = get_angled_range_data2(CORNERING[ci][0], CORNERING[ci][1])
    segs = get_pursuit_fits(CORNERING[ci][0], CORNERING[ci][1])

    prevsid = None

    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
    extent = (-30, 30, -10, 25)
    pdf_out = PdfPages("/tmp/endpoint_dist.pdf")
    for sid, d in groupby(data, 'session_id'):
        plt.figure()
        # What a mess! Only because rec_ungroup doesn't
        # work for nested types :(
        seg = [segs[i] for i in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)

        # 12 samples corresponds to about 1/60.0*12 = 0.2s
        # for "clean" pursuits.
        seg = seg[seg.n >= 12] 
        
        rows = d[d['ts'].searchsorted(seg.t0)]
        tp0 = np.vstack((rows['naksu_x'], rows['naksu_y'])).T
        rows = d[d['ts'].searchsorted(seg.t1)]
        tp1 = np.vstack((rows['naksu_x'], rows['naksu_y'])).T

        g0 = seg.d0 - tp0
        g1 = seg.d1 - tp1

        valid = np.isfinite(np.sum(g0, axis=1)) & np.isfinite(np.sum(g1, axis=1))
        g0 = g0[valid]
        g1 = g1[valid]
        
        clr0 = 'b'
        clr1 = 'r'
        plt.suptitle("Subject %s"%(str(sid)[-2:]))
        xyax = plt.subplot(1,1,1)

        plt.axvline(0, linestyle='dashed', color='black', alpha=0.5)
        plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
        
        pl, = xyax.plot(g0[:,0], g0[:,1], '.', color=clr0, alpha=0.5)
        pe, = xyax.plot(g1[:,0], g1[:,1], '.', color=clr1, alpha=0.5)
        
        plt.ylabel("Vert. gaze angle from TP")
        plt.xlabel("Horiz. gaze angle from TP")
        plt.gcf().legend((pl, pe), ('Fixation start', 'Fixation end'), 'upper right')

        divider = make_axes_locatable(xyax)

        xax = divider.append_axes("top", 1.5, pad=0.0, sharex=xyax)
        xrng = np.linspace(extent[0], extent[1], 300)
        dens0 = gaussian_kde(g0[:,0])
        dens1 = gaussian_kde(g1[:,0])
        xax.plot(xrng, dens0(xrng), color=clr0)
        xax.plot(xrng, dens1(xrng), color=clr1)
        xax.axvline(0, linestyle='dashed', color='black', alpha=0.5)
                
        yax = divider.append_axes("right", 1.5, pad=0.0, sharey=xyax)
        yrng = np.linspace(extent[2], extent[3], 300)
        dens0 = gaussian_kde(g0[:,1])
        dens1 = gaussian_kde(g1[:,1])
        yax.plot(dens0(yrng), yrng, color=clr0)
        yax.plot(dens1(yrng), yrng, color=clr1)
        yax.axhline(0, linestyle='dashed', color='black', alpha=0.5)

        plt.setp(xax.get_xticklabels() + yax.get_yticklabels(),
                     visible=False)
        xyax.set_aspect('equal')
        xyax.axis(extent)
        #plt.show()
        pdf_out.savefig(bbox_inches='tight')
    pdf_out.close()

def wstd(data, weights):
    weights = weights/np.sum(weights)
    weights = weights.flatten()
    wm = np.dot(data, weights)
    std = np.sqrt(np.dot(weights, (data-wm)**2))
    
    return std

def silverman_bandwidth(data, weights):
    d = np.size(data[0])
    if d > 1:
        stds = np.array([wstd(data[:,i], weights) for i in range(d)])
    else:
        stds = wstd(data, weights)

    n = len(data)
    return (4.0/(d + 2.0))**(1.0/(d+4.0))*n**(-1.0/(d+4.0))*stds

def fixed_std_gaussian_kde(means, stds=None, weights=None):
    if weights is None:
        weights = np.ones(len(means))
    weights = weights/np.sum(weights)
    
    if stds is None:
        stds = silverman_bandwidth(means, weights)
        print stds

    def density(y):
        # TODO: Figure out some broadcasting magic for this!
        cdens = np.zeros(len(y))
        normed = np.zeros(y.shape)
        density = cdens.copy()
        for i in range(means.shape[0]):
            #np.subtract(y, means[i], out=normed)
            normed[:] = y
            normed -= means[i]
            dns = scipy.stats.norm.pdf(normed, 0.0, stds)
            np.prod(dns.reshape(len(y), -1), axis=1, out=density)
            density *= weights[i]
            cdens += density
        return cdens
    
    return density

def segment_gaussian_kde_marginal(specs, dim, std):
    cdf = scipy.stats.norm.cdf
    dt = specs.t1 - specs.t0
    valid = dt != 0
    dt = dt[valid]
    specs = specs[valid]
    slopes = ((specs.d1[:,dim] - specs.d0[:,dim])/(dt))
    valid = slopes != 0 # Hack
    specs = specs[valid]
    slopes = slopes[valid]
    dt = dt[valid]
    
    d1 = specs.d1[:,dim]
    d0 = specs.d0[:,dim]

    if np.sum(~valid) != 0:
        print >>sys.stderr, "Marginal KDE throwing out %i segments due to a hack"%(np.sum(~valid))

    n = len(specs)

    w = 1.0/n
    
    #erf = approx_erf
    erf = scipy.special.erf
    sqrt = np.sqrt

    def density(val):
        cdens = []
        for i in range(n):
            #spec = specs[i]
            
            dens = (erf((d1[i]-val)/(sqrt(2)*std))-erf((d0[i] - val)/(sqrt(2)*std)))
            dens /= (2*slopes[i])
            #dens = cdf(val, d1[i], std) - cdf(val, d0[i], std)
            #dens /= -slopes[i]
            dens /= dt[i]
            dens *= w
            cdens.append(dens)
        return np.sum(cdens, axis=0)

    return density

def segment_gaussian_kde_2d(segs, sx, sy=None):
    dt = segs.t1 - segs.t0
    valid = (segs.n > 2) & (dt > 0)
    dt = dt[valid]
    segs = segs[valid]
    ax, ay = np.transpose((segs.d1 - segs.d0)/(dt).reshape(-1, 1))
    valid = (ax != 0) | (ay != 0) # Hack!

    if np.sum(~valid) != 0:
        print >>sys.stderr, "2d KDE throwing out %i segments due to a hack"%(np.sum(~valid))
    segs = segs[valid]
    ax = ax[valid]
    ay = ay[valid]
    
    t0 = segs.t0
    t1 = segs.t1

    bx = segs.d0[:,0] - ax*t0
    by = segs.d0[:,1] - ay*t0
    if sy is None:
        sy = sx
    
    n_segs = float(len(ay))

    def density(d):
        lik = np.zeros(len(d))
        for i in range(len(ay)):
            lik += segment_likelihood_2d(d, t0[i], t1[i],
                ax[i], ay[i], bx[i], by[i],
                sx, sy)
        return lik/n_segs
    
    return density

def approx_erf(x):
    coeff = [1.0, 0.278393, 0.230389, 0.000972, 0.078108][::-1]
    return np.sign(x)*(1 - 1.0/(np.polyval(coeff, np.abs(x)))**4)

def segment_likelihood_2d(d, t0, t1, ax, ay, bx, by, sx, sy):
    # Maxima: integrate(normal_pdf(ax*t+bx, mx, sx)*normal_pdf(ay*t+by, my, sy), t, t0, t1)
    # plus a lot of simplification.
    # TODO: Looks a lot like some kind of definite integral of gaussian pdf
    # TODO: Optimize like hell!
    exp = np.exp
    sqrt = np.sqrt
    erf = scipy.special.erf
    #erf = approx_erf
    
    mx = d[:,0]
    my = d[:,1]
    
    dy = ay*by-ay*my
    dx = ax*bx-ax*mx
    var = sx**2*ay**2+sy**2*ax**2

    e = exp(-(ax*my-ay*mx-ax*by+ay*bx)**2/(2*(var)))
    erfdiff = erf(((var)*t1+(dy)*sy**2+(dx)*sx**2)/(sqrt(2)*sx*sy*sqrt(var)))
    erfdiff -= erf(((var)*t0+(dy)*sy**2+(dx)*sx**2)/(sqrt(2)*sx*sy*sqrt(var)))
    lik = e*erfdiff / (2**(3/2.0)*sqrt(np.pi)*sqrt(var))
    lik /= (t1 - t0)
    return lik

def histogram_hdr(hist, bin_size, mass_limits):
    # Not very fast!
    ordered = np.sort(hist.flatten())[::-1]
    cum_dens = np.cumsum(ordered*bin_size)
    
    density_limits = np.searchsorted(cum_dens, mass_limits)
    invalid = density_limits >= len(ordered)
    invalid_values = [np.nan] * np.sum(invalid)
    values = ordered[density_limits[~invalid]]
    return np.array(list(values) + invalid_values)

def density_figure((X, Y, xydens), (xrng, xdens), (yrng, ydens), hdr_levels):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    extent = xrng[0], xrng[-1], yrng[0], yrng[-1]
    binsize = (
        np.abs(np.diff(extent[:2]))/len(xrng),
        np.abs(np.diff(extent[2:]))/len(yrng),
        )


    xyax = plt.subplot(1,1,1)
    levels = histogram_hdr(xydens, np.prod(binsize), hdr_levels)

    xyax.imshow(np.rot90(xydens), extent=extent, cmap=plt.cm.gray_r)
    xyax.axvline(0, color='black', linestyle='dashed', alpha=0.5)
    xyax.axhline(0, color='black', linestyle='dashed', alpha=0.5)
   
    plt.contour(X, Y, xydens, colors='Black', linestyles='solid', alpha=0.5, levels=levels)

    plt.locator_params(axis='both', prune='both')

    divider = make_axes_locatable(xyax)
    
    xax = divider.append_axes("top", 1.5, pad=0.0, sharex=xyax)
    xax.plot(xrng, xdens, color='Black')
    xax.axvline(0, linestyle='dashed', color='black', alpha=0.5)

    levels = histogram_hdr(xdens, binsize[0], hdr_levels)
    for l in levels:
        in_level = xdens > l
        xax.fill_between(xrng, xdens, where=in_level, color='black', alpha=0.25)
    plt.ylim(0, None)
    plt.locator_params(axis='y', nbins=5, prune='lower')
    
    yax = divider.append_axes("right", 1.5, pad=0.0, sharey=xyax)
    yax.plot(ydens, yrng, color='black')
    yax.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    
    levels = histogram_hdr(ydens, binsize[1], hdr_levels)
    for l in levels:
        in_level = ydens >= l
        yax.fill_betweenx(yrng, ydens, where=in_level, color='black', alpha=0.25)
    plt.xlim(0, None)
    plt.locator_params(axis='x', nbins=5, prune='lower')
    
    
    plt.setp(xax.get_xticklabels() + yax.get_yticklabels(),
                    visible=False)
    xyax.set_aspect('equal')
    xyax.axis(extent)

    return (xyax, xax, yax)

def stripped_density_figure((X, Y, xydens), (xrng, xdens), (yrng, ydens), hdr_levels, color='Black', 
                            xax=None, yax=None, xyax=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    extent = xrng[0], xrng[-1], yrng[0], yrng[-1]
    binsize = (
        np.abs(np.diff(extent[:2]))/len(xrng),
        np.abs(np.diff(extent[2:]))/len(yrng),
        )

    if xyax is None:
        xyax = plt.subplot(111)
    levels = histogram_hdr(xydens, np.prod(binsize), hdr_levels)

    #xyax.imshow(np.rot90(xydens), extent=extent, cmap=plt.cm.gray_r)
    xyax.axvline(0, color='black', linestyle='dashed', alpha=0.5)
    xyax.axhline(0, color='black', linestyle='dashed', alpha=0.5)
   
    #plt.contour(X, Y, xydens, colors=color, linestyles='solid', alpha=0.5, levels=levels)
    xyax.contour(X, Y, xydens, colors=color, linestyles='solid', alpha=0.5, levels=levels[1:2])
    plt.locator_params(axis='both', prune='both')

    if xax is None:
        #xax = plt.subplot(2,2,1, sharex=xyax)
        divider = make_axes_locatable(xyax)
        xax = divider.append_axes("top", 1.5, pad=0.0, sharex=xyax)
    xax.plot(xrng, xdens, color=color)
    xax.autoscale()
    xax.axvline(0, linestyle='dashed', color='black', alpha=0.5)

    #levels = histogram_hdr(xdens, binsize[0], hdr_levels)
    #for l in levels:
    #    in_level = xdens > l
    #    xax.fill_between(xrng, xdens, where=in_level, color='black', alpha=0.25)
    plt.ylim(0, None)
    plt.locator_params(axis='y', nbins=5, prune='lower')
    
    if yax is None:
        #yax = plt.subplot(2,2,4, sharey=xyax)
        yax = divider.append_axes("right", 1.5, pad=0.0, sharey=xyax)
    yax.plot(ydens, yrng, color=color)
    yax.autoscale()
    yax.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    
    #levels = histogram_hdr(ydens, binsize[1], hdr_levels)
    #for l in levels:
    #    in_level = ydens >= l
    #    yax.fill_betweenx(yrng, ydens, where=in_level, color='black', alpha=0.25)
    
    plt.xlim(0, None)
    plt.locator_params(axis='x', nbins=5, prune='lower')
     
    plt.setp(xax.get_xticklabels() + yax.get_yticklabels(),
                    visible=False)
    
    xyax.set_aspect('equal')
    xyax.axis(extent)
    
    return (xyax, xax, yax)


def gaze_location_density():
    from segreg import piecewise_linear_regression, \
        piecewise_linear_regression_pseudo_em,\
        regression_interpolator, slope_density, slope_stats

    ci = [1,3]
    is_control = False

    data, segs = get_pooled(ci) 
    data, segs = filter_by_treatment(data, segs, is_control)
    data.sort(order=['session_id', 'ts'])

    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
    extent = (-30, 30, -10, 25)
    hdr_levels = [0.25, 0.5, 0.75]
    #extent = (-30, 30, -9, 90)

    txt = "free" if is_control else "tp"
    pdf_out = PdfPages("/tmp/gaze_density_"+str(ci)+"_"+txt+".pdf")
    
    std = 1.0
    binsize = 0.1

    xrng = np.arange(extent[0], extent[1], binsize)
    yrng = np.arange(extent[2], extent[3], binsize)
    X, Y = np.mgrid[extent[0]:extent[1]:binsize, extent[2]:extent[3]:binsize]
    

    agg = dict(
        xydens=np.zeros(X.shape),
        xdens=np.zeros(xrng.shape),
        ydens=np.zeros(yrng.shape)
        )
    modes = {}


    for sid, d in groupby(data, 'session_id'):
        print sid
        # What a mess! Only because rec_ungroup doesn't
        # work for nested types :(
        seg = [segs[i] for i in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)
        #seg = seg[:100]
        
        rows = d[ d['ts'].searchsorted(seg.t0) ]
        tp0 = np.vstack((rows['naksu_x'], rows['naksu_y'])).T
        rows = d[ d['ts'].searchsorted(seg.t1) ]
        tp1 = np.vstack((rows['naksu_x'], rows['naksu_y'])).T
        valid = np.isfinite(np.sum(tp0, axis=1)) & np.isfinite(np.sum(tp1, axis=1))
        tp0 = tp0[valid]
        tp1 = tp1[valid]
        seg = seg[valid]
        
        seg.d0 -= tp0
        seg.d1 -= tp1
        g0 = seg.d0
        g1 = seg.d1
        
        densf = segment_gaussian_kde_2d(seg, std)
        coords = np.vstack((X.flatten(), Y.flatten())).T
        xydens = densf(coords).reshape(X.shape)
        agg['xydens'] += xydens
        
        def negdens(d):
            return -densf(d.reshape(1, 2))
        
        print "Finding mode"
        mode = np.argmax(xydens)
        mode = np.array([X.flatten()[mode], Y.flatten()[mode]])
        bounds = np.transpose([mode - binsize/2.0, mode + binsize/2.0])
        mode = scipy.optimize.minimize(negdens, mode,
            method='COBYLA', tol=1e-6, bounds=bounds).x
        modeval = densf(mode.reshape(1, 2))[0]
        modes[sid] = mode
        print "found", mode

        dens = segment_gaussian_kde_marginal(seg, 0, std)
        xdens = dens(xrng)
        agg['xdens'] += xdens
                    
        dens = segment_gaussian_kde_marginal(seg, 1, std)
        ydens = dens(yrng)
        agg['ydens'] += ydens

        #continue
        plt.figure()
        plt.suptitle("Subject %s"%(str(sid)[-2:]))

        xyax, xax, yax = density_figure((X, Y, xydens), (xrng, xdens), (yrng, ydens), hdr_levels)
        xyax.scatter(mode[0], mode[1], color='white', alpha=0.8)
        xyax.set_ylabel("Vert. gaze angle from TP")
        xyax.set_xlabel("Horiz. gaze angle from TP")
        #plt.show()
        pdf_out.savefig(bbox_inches='tight')
    
    plt.figure()
    n = len(modes)
    
    xyax, xax, yax = density_figure(
        (X, Y, agg['xydens']/n),
        (xrng, agg['xdens']/n),
        (yrng, agg['ydens']/n),
        hdr_levels)
    
    for sid, mode in modes.items():
        print sid, mode
    
    modes = np.array(modes.values())
    #plt.scatter(modes[:,0], modes[:,1], color='white', alpha=0.8)
    xyax.scatter(modes[:,0], modes[:,1], s=6, facecolors='white', edgecolors='black', alpha=0.8)
    xyax.set_ylabel("Vert. gaze angle from TP")
    xyax.set_xlabel("Horiz. gaze angle from TP")

    pdf_out.savefig(bbox_inches='tight')
    
    
    plt.figure()
    
    plt.axvline(0, color='black', linestyle='dashed', alpha=0.5)
    plt.axhline(0, color='black', linestyle='dashed', alpha=0.5)
    plt.scatter(modes[:,0], modes[:,1], color='black')
    plt.ylabel("Vert. gaze angle from TP")
    plt.xlabel("Horiz. gaze angle from TP")


    pdf_out.savefig(bbox_inches='tight')
    pdf_out.close()

def seg_in_aoi(seg, data):
    #points = [ (data[data['ts'] == seg[x]) for x in seg ]
    points = []
    sx = []
    sy = []
    seg = np.array(seg)
    for s in seg.flatten():
        print len(s)
        sx.append(s[2][0])
        sy.append(s[2][1])
        point = data[data['ts'] == s[0]]
        points.append(point)
    
    points = np.array(points)
    nx, ny = px2heading(points[:]['naksu_x'], points[:]['naksu_y'])

    dists = []
    for i, s in enumerate(seg):
        d = np.sqrt( np.abs(sx[i]-nx[i])**2 + np.abs(sy[i]-ny[i])**2 )
        dists.append(d)
    
    #print sx #dists, len(dists)
    return False

def angle_density():
    ci = [1,3]
    is_control = False
 
    data, segs = get_pooled(ci)   
    data, segs = filter_by_treatment(data, segs, is_control)
 
    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')

    txt = "free" if is_control else "tp"    
    pdf_out = PdfPages('/tmp/angle_dist_'+str(ci)+'_'+txt+'.pdf')

    std = 0.1
    rng = np.arange(-np.pi, np.pi, 0.01)

    all_angles = []
    all_weights = []

    for sid, d in groupby(data, 'session_id'):
        print sid
        # What a mess! Only because rec_ungroup doesn't
        # work for nested types :(
        seg = [segs[i] for i in np.flatnonzero(groups.session_id == sid)]

        seg = np.hstack(seg).view(np.recarray)
        
        #seg_to_data_i = d['ts'].searchsorted(seg.t0)
        #seg_d = d[seg_to_data_i]
        #print seg.d0[:,0]
        #dist_to_tp = np.sqrt((seg_d['naksu_x'] - seg.d0[:,0])**2 + (seg_d['naksu_y'] - seg.d0[:,1])**2)
        #from_tp = dist_to_tp < 3
        
        # Trim out "saccades"
        seg = seg[(seg.n > 12)]
        #& ~from_tp]

        dt = seg.t1 - seg.t0
        valid = dt != 0
        seg = seg[valid]
        dt = dt[valid].reshape(-1, 1)

        #slopes = (seg.d1 - seg.d0)/dt
        d = seg.d1 - seg.d0
        angles = np.arctan2(d[:,0], d[:,1])

        
        plt.figure()
        plt.suptitle("Subject %s"%(str(sid)[-2:]))
        ax = plt.subplot(1,1,1, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        
        #densf = fixed_std_gaussian_kde(angles, std, dt)
        #dens = densf(rng)
        
        all_angles.extend(angles)
        all_weights.extend(dt/float(len(dt)))

        bins = 60
        
        if (len(angles) == 0): continue
        plt.hist(angles, bins=bins, weights=dt, color='black', normed=True, alpha=0.5)

        pdf_out.savefig(bbox_inches='tight')
        #plt.plot(rng, dens)
    
    plt.figure()
    ax = plt.subplot(1,1,1, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    bins = 180
    plt.hist(np.array(all_angles), weights=np.array(all_weights),
        bins=bins, color='black', normed=True, alpha=0.5)
    plt.ylim((0,1.4))
     
    pdf_out.savefig(bbox_inches='tight')
    
    pdf_out.close()

def horiz_delta_vs_half_yaw():
    #import statsmodels.api as sm
    ci = 1
    data = get_angled_range_data2(CORNERING[ci][0], CORNERING[ci][1])
    segs = get_pursuit_fits(CORNERING[ci][0], CORNERING[ci][1])
    
    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
    
    medians = {}
    coeffs = []
    for sid, d in groupby(data, 'session_id'):
        # What a mess! Only because rec_ungroup doesn't
        # work for nested types :(
        seg = [segs[i] for i in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)
        
        # Trim out "saccades"
        seg = seg[(seg.t1 - seg.t0) > 0.1]

        dt = seg.t1 - seg.t0
        valid = dt != 0
        seg = seg[valid]
        dt = dt[valid].reshape(-1, 1)
        slopes = (seg.d1 - seg.d0)/dt
        
        yaw = toyota_yaw_rate_in_degrees(d['c_yaw'])
        mean_yaw = []
        for s in seg:
            span = np.searchsorted(d['ts'], [s.t0, s.t1])
            myaw = np.mean(yaw[slice(*span)])
            mean_yaw.append(myaw)
        
        mean_yaw = np.array(mean_yaw)

        coeff = scipy.stats.spearmanr(-mean_yaw, slopes[:,0])[0]
        coeffs.append(coeff)
        print sid, coeff
        
        #plt.figure()
        #lap = d[d['lap'] == 10]
        #plt.plot(lap['ts'],lap['c_yaw'], '-k')
        #plt.show()
        
        #plt.scatter(mean_yaw/2.0, slopes[:,0])
        #plt.scatter(mean_yaw/2.0, slopes[:,0])
        diff = slopes[:,0] + mean_yaw/2.0
        medians[sid] = np.median(diff)

    mcoeff = np.median(coeffs)
    mdiffs = medians.values()
    print "median rs: %f" % mcoeff 
    print "Shapiro ", scipy.stats.shapiro(mdiffs)
    print "CI", t_ci(mdiffs)

@memorize
def gaze_delta_density(is_control=False):
    from segreg import piecewise_linear_regression, \
        piecewise_linear_regression_pseudo_em,\
        regression_interpolator, slope_density, slope_stats

    ci = [1,3]
    #is_control = False

    data, segs = get_pooled(ci) 
    data, segs = filter_by_treatment(data, segs, is_control)

    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
  
    extent = (-25, 20, -20, 20)
    hdr_levels = [0.25, 0.5, 0.75]
    #extent = (-30, 30, -9, 90)
    txt = "free" if is_control == True else "tp"
    pdf_out = PdfPages("/tmp/delta_density_"+str(ci)+"_"+txt+".pdf")

    
    std = 1.5
    binsize = 0.1

    xrng = np.arange(extent[0], extent[1], binsize)
    yrng = np.arange(extent[2], extent[3], binsize)
    X, Y = np.mgrid[extent[0]:extent[1]:binsize, extent[2]:extent[3]:binsize]
        
    xlabel = "Horiz. pursuit velocity (deg/s)"
    ylabel = "Vert. pursuit velocity (deg/s)"

    agg = dict(
        xydens=np.zeros(X.shape),
        xdens=np.zeros(xrng.shape),
        ydens=np.zeros(yrng.shape)
        )
    modes = {}

    for sid, d in groupby(data, 'session_id'):
        # What a mess! Only because rec_ungroup doesn't
        # work for nested types :(
        seg = [segs[i] for i in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)
        
        # Trim out "saccades"
        seg = seg[(seg.t1 - seg.t0) > 0.1]

        dt = seg.t1 - seg.t0
        valid = dt != 0
        seg = seg[valid]
        dt = dt[valid].reshape(-1, 1)
        

        slopes = (seg.d1 - seg.d0)/dt
        
        
        xydensf = fixed_std_gaussian_kde(slopes, weights=dt)
        coords = np.vstack((X.flatten(), Y.flatten())).T
        xydens = xydensf(coords).reshape(X.shape)
        
        def negdens(d):
            return -xydensf(d.reshape(1, 2))

        print sid
        print "Finding mode"
        mode = np.argmax(xydens)
        mode = np.array([X.flatten()[mode], Y.flatten()[mode]])
        bounds = np.transpose([mode - binsize/2.0, mode + binsize/2.0])
        mode = scipy.optimize.minimize(negdens, mode,
            method='COBYLA', tol=1e-6, bounds=bounds).x
        modeval = xydensf(mode.reshape(1, 2))[0]
        modes[sid] = mode
        print "found", mode
        
        #xdens = np.sum(xydens, axis=1)
        #ydens = np.sum(xydens, axis=0)
        
        xdens = fixed_std_gaussian_kde(slopes[:,0], weights=dt)(xrng)
        ydens = fixed_std_gaussian_kde(slopes[:,1], weights=dt)(yrng)

        agg['xydens'] += xydens
        agg['xdens'] += xdens
        agg['ydens'] += ydens

        plt.figure()
        plt.suptitle("Subject %s"%(str(sid)[-2:]))
        xyax, xax, yax = density_figure((X, Y, xydens), (xrng, xdens), (yrng, ydens), hdr_levels)
        
        xyax.scatter(mode[0], mode[1], color='white', alpha=0.8)
        
        xyax.set_xlabel(xlabel)
        xyax.set_ylabel(ylabel)

        pdf_out.savefig(bbox_inches='tight')
        plt.close()
        #xdens = fixed_std_gaussian_kde(slopes[:,1], std, dt)
        #xdens = fixed_std_gaussian_kde(slopes[:,1], std, dt)
        #plt.imshow(np.rot90(xydens), extent=extent)
        
    plt.figure()
    n = len(modes)
    
    xyax, xax, yax = density_figure(
        (X, Y, agg['xydens']/n),
        (xrng, agg['xdens']/n),
        (yrng, agg['ydens']/n),
        hdr_levels)
    
    for sid, mode in modes.items():
        print sid, mode
 
    modes = np.array(modes.values())
    #plt.scatter(modes[:,0], modes[:,1], color='white', alpha=0.8)
    xyax.scatter(modes[:,0], modes[:,1], s=6, facecolors='white', edgecolors='black', alpha=0.8)
    xyax.set_ylabel(ylabel)
    xyax.set_xlabel(xlabel)

    pdf_out.savefig(bbox_inches='tight')
    
    
    plt.figure()
    
    plt.axvline(0, color='black', linestyle='dashed', alpha=0.5)
    plt.axhline(0, color='black', linestyle='dashed', alpha=0.5)
    plt.scatter(modes[:,0], modes[:,1], color='black')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
   
    pdf_out.savefig(bbox_inches='tight')
    plt.close()
    pdf_out.close()

    return modes, (X,Y,xrng,yrng,agg,hdr_levels)

def gaze_delta_density_plot_everything():

    from tru.studies.ramppi11.analyze_opa import gaze_delta_density as gd11    
    from tru.studies.ramppi11.analyze_opa import get_pooled as gp11

    m13free, d13free = gaze_delta_density(is_control=True)
    m13tp, d13tp = gaze_delta_density(is_control=False)
    m11, d11 = gd11()
    modes = (m13free, m13tp, m11)
    densities = (d13free, d13tp, d11)

    pdf_out = PdfPages('/tmp/delta_density_all.pdf')
    xlabel = "Horiz. pursuit velocity (deg/s)"
    ylabel = "Vert. pursuit velocity (deg/s)"

    colors = ['green', 'red', 'blue']
    xyax = xax = yax = None

    # half yaw rates
    data11, segs11 = gp11([1,3])
    data13, segs13 = get_pooled([1,3])
    #data13tp, segs13tp = filter_by_treatment(data13, segs13, control=False)
    #data13free, segs13free = filter_by_treatment(data13, segs13, control=True)
    yaw11 = toyota_yaw_rate_in_degrees(np.mean(data11['c_yaw'])) / 2.0
    yaw13 = toyota_yaw_rate_in_degrees(np.mean(data13['c_yaw'])) / 2.0
    myaw = np.mean([yaw11, yaw13])
    #yaw13tp = toyota_yaw_rate_in_degrees(np.mean(data13tp['c_yaw'])) / 2.0
    #yaw13free = toyota_yaw_rate_in_degrees(np.mean(data13free['c_yaw'])) / 2.0
    #print yaw11, yaw13tp, yaw13free

    plt.figure()
    for i, m in enumerate(modes):
        X, Y, xrng, yrng, agg, hdr_levels = densities[i]
        n = len(m)
        xyax, xax, yax = stripped_density_figure(
            (X, Y, agg['xydens']/n),
            (xrng, agg['xdens']/n),
            (yrng, agg['ydens']/n),
            hdr_levels, color=colors[i],
            xax=xax, yax=yax, xyax=xyax)
        xyax.scatter(m[:,0], m[:,1], s=6, facecolors=colors[i], 
                    edgecolors='black', linewidths=0, alpha=0.8)
    
    xyax.axvline(-myaw, color='black', linestyle='dotted')
    xyax.set_ylabel(ylabel)
    xyax.set_xlabel(xlabel)
    pdf_out.savefig(bbox_inches='tight')
    plt.close()
    pdf_out.close()

def gaze_delta_density_free_vs_tp():
    from tru.studies.ramppi11.analyze_opa import gaze_delta_density as gd11
    c = gd11()
    f = gaze_delta_density(is_control=True)
    t = gaze_delta_density(is_control=False)
    diff = f[0]-t[0]
    #diff = diff[:,0].reshape(-1,1)
    n = len(diff)
    #co = np.cov(diff.T, ddof=1).reshape(1,1)
    #cov = np.asmatrix(scipy.linalg.inv(co))
    #print cov
    cov = np.asmatrix(scipy.linalg.inv(np.cov(diff.T, ddof=1)))
    mean = np.asmatrix(np.mean(diff, axis=0)).T
    tsq = n * mean.T * cov * mean
    dim = 2.0
    fstat = (n-dim) / (dim*(n-1)) * tsq
    print tsq, fstat
    fd = scipy.stats.f(dim,n-dim)
    print fd.sf(fstat)

    #print scipy.stats.ttest_rel(c[:,0],t[:,0]) 
    #print scipy.stats.ttest_rel(c[:,1],t[:,1]) 

    """
    x = c[:,0]-t[:,0]
    y = c[:,1]-t[:,1]
    plt.figure()
    plt.title('Control-TP')
    plt.plot(x,y, '.k')
    xlabel = "Horiz. pursuit velocity (deg/s)"
    ylabel = "Vert. pursuit velocity (deg/s)"
    plt.show()
    plt.close()
    """


def pursuit_location_vs_slope():
    from segreg import piecewise_linear_regression, \
            piecewise_linear_regression_pseudo_em,\
            regression_interpolator, slope_density, slope_stats

    ci = 1
    data = data[data['g_direction_q'] > 0.2]
    
    sid = None
    for d in foreach(data, ['session_id', 'lap']):
        if sid != d['session_id'][0] and sid is not None:
            plt.title(str(sid))
            plt.axis([-40, 40, -5, 20])
            plt.gca().set_aspect('equal')
            plt.show()
        sid = d['session_id'][0]

        h = d['scenecam_x']
        p = d['scenecam_y']
        t = d['ts'] - d['ts'][0]
        g = np.vstack((h, p)).T
        
        params = ((1.5, 1.5), 0.5)
        splits, valid, winner, params =\
            memorize(piecewise_linear_regression_pseudo_em)(
                t, g, *params)
        
        vt = t[valid]
        vg = g[valid]
        #stats = slope_stats(splits, vt, vg)
        
        #plt.figure()
        #stdstr = ",".join("%.3f"%p for p in params[0])
        #plt.suptitle("%i/%i (t0: %i, noise %s fix_dur %.3f)"%(
        #    sid, lap, d['ts'][0], stdstr, params[1]))
        
        fit = regression_interpolator(splits, vt, vg)

        for i in range(0, len(fit.x)-1, 2):
            ts = fit.x[i:i+2]
            if ts[1] - ts[0] < 0.1:
                continue
            rows = d[np.searchsorted(t, ts)]
            tpx, tpy = rows['naksu_x'], rows['naksu_y']

            
            gx, gy = fit.y[:, i:i+2]
            
            gx -= tpx
            gy -= tpy

            dx = gx[0] - gx[1]
            dy = gy[0] - gy[1]
            
            #plt.plot(gx, gy, color='black', alpha=0.3)
            plt.plot(gx[0], gy[0], 'b.', alpha=0.3)
            plt.plot(gx[1], gy[1], 'r.', alpha=0.3)
            #plt.arrow(gx[0], gy[0], dx, dy)
    plt.title(str(sid))
    plt.axis([-40, 40, -5, 20])
    plt.gca().set_aspect('equal')
    plt.show()


def plot_segmentations():
    from segreg import piecewise_linear_regression, \
            piecewise_linear_regression_pseudo_em,\
            segmentation_to_table

    ci = 1
    data = get_angled_range_data2(CORNERING[ci][0], CORNERING[ci][1])
    
    pdf_out = PdfPages('/tmp/gazefit.pdf')

    for od in foreach(data, ['session_id', 'lap']):
        plt.figure()
        sid = od['session_id'][0]
        lap = od['lap'][0]

        goodsei = od['g_direction_q'] > 0.2
        d = od[goodsei]

        print sid, lap
            
        h = d['scenecam_x']
        p = d['scenecam_y']
        t = d['ts'] - d['ts'][0]
        


        g = np.vstack((h, p)).T
        
        params = ((1.5, 1.5), 0.5)
        splits, valid, winner, params =\
            memorize(piecewise_linear_regression_pseudo_em)(
                t, g, *params)
        
        vt = t[valid]
        vg = g[valid]
        
        segs = segmentation_to_table(splits, vt, vg)
        
        plt.suptitle("Subject %s, lap %i, noise std %s"%(
            str(sid)[-2:], lap, "h: %.3f, v: %.3f"%tuple(params[0])))
        
        def dimplot(dim):
            sename = ['scenecam_x', 'scenecam_y']
            tpname = ['naksu_x', 'naksu_y']
            plt.plot(od['ts'] - od['ts'][0], od[tpname[dim]], color='black', alpha=0.5)

            plt.plot(vt, vg[:,dim], 'k.', alpha=0.2)
            plt.plot(t[~valid], g[~valid,dim], 'b.', alpha=0.2)
            plt.plot(od[~goodsei]['ts']-d['ts'][dim], od[~goodsei][sename[dim]], 'y.', alpha=0.2)
            seg = segs[segs.t1 - segs.t0 > 0.1]
            plt.plot([seg.t0, seg.t1], [seg.d0[:,dim], seg.d1[:,dim]], color='red',
                linewidth=2, alpha=0.8)
            seg = segs[segs.t1 - segs.t0 <= 0.1]
            plt.plot([seg.t0, seg.t1], [seg.d0[:,dim], seg.d1[:,dim]], color='red',
                linewidth=2, alpha=0.8, linestyle='dotted')


        ax = plt.subplot(2,1,1)
        
        dimplot(0)
        plt.ylabel("Horiz. gaze angle (deg)")
        plt.ylim(0, 40)
        plt.xlim(0, 18)
        
        plt.subplot(2,1,2, sharex=ax)
        dimplot(1)
        plt.ylabel("Vert. gaze angle (deg)")
        plt.xlabel("Cornering time (s)")
        plt.ylim(-15, 25)
        plt.xlim(0, 18)
        
        #plt.show(); continue
        pdf_out.savefig(bbox_inches='tight')
    pdf_out.close()


def hotelling_ellipse(data):
    mean = np.mean(data, axis=0)
    print mean
    W = np.cov(data.T, ddof=1).T
    n = float(len(data))
    p = float(len(mean))
    (an, bn), (axes) = np.linalg.eig(W)
    ang = -np.arctan(axes[0][1]/axes[0][0])
    ev = np.array((an, bn))
    
    def getit(t, level):
        f = scipy.stats.f.ppf(level, p, n-p)
        t_to_f = p*(n - 1)/(n*(n - p))
        l = np.sqrt(ev*t_to_f*f)

        x = mean[0] + l[0]*np.cos(t)*np.cos(ang) - l[1]*np.sin(t)*np.sin(ang)
        y = mean[1] + l[0]*np.cos(t)*np.sin(ang) + l[1]*np.sin(t)*np.cos(ang)
        return x, y
    return getit

# Cached 'cause they are a bit slow to compute
c1_gaze_modes = np.array([
[ 6.99085229, 6.3865011 ],
[ 1.29320534, 2.83864224],
[ 2.42364528, 3.45623079],
[ 1.37843297, 3.97108062],
[ 2.07258402, 3.65412031],
[ 1.37104515, 1.48497186],
[ 4.26994542, 3.59808061],
[ 1.10531991, 3.06232151],
[ 4.44993965, 3.30082162],
[-4.8175737,  2.20514463],
[ 0.55961094, 3.77210797],
[-0.25042814, 2.64676785],
[ 2.57432312, 4.19425749],
[ 0.47393129, 4.7116272 ],
[-2.73753869, 3.09624288],
[ 5.46911462, 4.62597864],
[ 2.97400622, 3.53413117],
])


c1_delta_modes = np.array([
 [-7.12131182, -0.84937597],
 [-7.81271557, -0.98065128],
 [-6.86406178, -1.66540003],
 [-5.90239452, -2.62825119],
 [-7.40496816, -2.11133353],
 [-6.31692131, -1.19766939],
 [-6.17776772, -0.88219578],
 [-8.39538434, -2.63338688],
 [-7.96484461, -1.01482004],
 [-6.63790088, -2.21085101],
 [-8.28147133, -2.04041112],
 [-7.90222433, -0.99504704],
 [-7.88664639, -1.20583918],
 [-5.2408938 , -1.77881317],
 [-7.55280479, -1.75450008],
 [-6.88266931, -2.04956501],
 [-5.48264619, -2.60811515],
])

# missing 2,4,8,9,10,20 ?
# order by date
c2_delta_modes = np.array([
[-6.09059192, -2.14614134],
[-5.07739785, -1.76681922],
[-6.81345248, -2.95154618],
[-7.00624548, -2.57402529],
[-7.88262961, -0.59913177],
[-8.49481663, -0.86846655],
[-8.65733117, -1.6165484 ],
[-6.99363262, -1.71549632],
[-8.16070745, -1.3740833 ],
[-8.71235062, -1.05404838],
[-7.09256714, -1.14736588],
[-8.97594679, -0.5084806 ],
[-7.3812893,  -1.96110881],
[-7.35813775, -1.83746673],
[-6.57783255, -0.76251256]
])

#missing 4,8,9,20
#order (seemingly) random 
c3_delta_modes = np.array([
 [-7.30619455, -2.11363571],
 [-7.73224688, -1.60025428],
 [-6.94783859, -0.96338522],
 [-6.06213273, -2.61793268],
 [-8.16819779, -2.19522806],
 [-8.03198563, -1.56029198],
 [-6.85716605, -1.28027765],
 [-8.69231443, -2.7630825 ],
 [-8.262576  , -1.44883395],
 [-6.41811214, -2.54368683],
 [-7.81896956, -1.95492263],
 [-7.00005753, -0.69692548],
 [-8.06291071, -1.25824585],
 [-6.21884446, -1.51016074],
 [-8.05864165, -3.18469216],
 [-7.55327979, -1.33281182],
 [-8.32554447, -3.53280075]
])

def confidence_region_plot(data=c1_delta_modes):
    plt.scatter(data[:,0], data[:,1], color='black')
    ellipser = hotelling_ellipse(data)
    
    trng = np.linspace(0, 2*np.pi, 100)
    plt.plot(*ellipser(trng, 0.95), color='black', alpha=0.5)
    plt.plot(*ellipser(trng, 0.99), color='black', alpha=0.5)
    plt.plot(*ellipser(trng, 0.999), color='black', alpha=0.5)

    ci = 3
    bdata = get_angled_range_data2(CORNERING[ci][0], CORNERING[ci][1])
    yawhalf = -np.mean(toyota_yaw_rate_in_degrees(bdata['c_yaw'])) / 2
    print yawhalf
    
    plt.axvline(0, color='black', alpha=0.5, linestyle='dashed')
    plt.axvline(yawhalf, color='red', alpha=0.5, linestyle='dashed')
    plt.axhline(0, color='black', alpha=0.5, linestyle='dashed')

    plt.xlabel("Horiz. gaze velocity (deg/s)")
    plt.ylabel("Vert. gaze velocity (deg/s)")
    #plt.xlabel("Horiz. gaze position (deg)")
    #plt.ylabel("Vert. gaze position (deg)")

    plt.axis([-10, 2, -5, 2])
    plt.gca().set_aspect('equal')
    
    plt.show()

def tp_dist_vs_pursuit_angle():
    cis = [1,2,3]
    all_data = []
    all_segs = []
    for ci in cis:
        data = get_angled_range_data2(CORNERING[ci][0], CORNERING[ci][1])
        segs = get_pursuit_fits(CORNERING[ci][0], CORNERING[ci][1])
        all_data.append(data)
        all_segs.append(segs)
    data = np.hstack(all_data)
    segs = np.vstack(all_segs)
    data.sort(order=['session_id', 'ts'])
    
    #segs = rec_degroup(segs, ['session_id', 'lap'])
    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
    
    data = append_field(data, 'dist_from_tp', np.nan)
    all_angles = []
    all_dists = []
    for sid, i in groupby_i(data, 'session_id'):
        d = data[i]
        print sid
        seg = [segs[k] for k in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)
        seg = seg[seg.n > 12]
        seg_to_data_i = d['ts'].searchsorted(seg.t0)
        seg_d = d[seg_to_data_i]
        
        dist_to_tp = np.sqrt((seg_d['naksu_x'] - seg.d0[:,0])**2 + (seg_d['naksu_y'] - seg.d0[:,1])**2)
        data['dist_from_tp'][i] = dist_to_tp

        delta = seg.d1 - seg.d0
        angles = np.arctan2(delta[:,0], delta[:,1])
        all_angles.extend(angles)
        all_dists.extend(dist_to_tp)
        

    angles_deg = np.degrees(all_angles)
    from tru.route import angle_difference
    dist_from_180 = angle_difference(180,angles_deg)

    plt.figure()
    plt.plot(all_dists, dist_from_180, ',k')
    plt.xlabel('distance from TP (deg)')
    plt.ylabel('absolute angular difference from downward vertical axis (deg)')
    plt.show()
    
def conditions_ttest():
    from tru.studies.ramppi11.analyze_opa import get_pooled as get11
    ci = [1,3]
    data, segs = get_pooled(ci)
    control = get11(ci)
    tp = filter_by_treatment(data, segs, control=False)
    free = filter_by_treatment(data, segs, control=True)
    conditions = [tp, free, control]
         
    def get_means(condition):
        data, segs = condition
        sdata = []
        for sid, d in groupby(data, 'session_id'):
            seg = None
            seg = [s[1] for s in segs if s[0][0] == sid]
            #for s in segs:
            #    if (s[0][0] == sid):
            #        seg.append(s)
            if seg is None: continue
            seg = np.hstack(seg).view(np.recarray)
            seg = seg[ seg.n > 12 ]
            start = seg.t0
            stop = seg.t1
            dt = stop - start
            gdiff = seg.d1 - seg.d0
            gdiff = gdiff[:,0]
            gspeed = gdiff / dt
            yaw = []
            for s1, s2 in zip(start, stop):
                fix = d[ (s1 <= d['ts']) & (s2 >= d['ts']) ]
                y = toyota_yaw_rate_in_degrees(np.mean(fix['c_yaw']))
                yaw.append(y)
            test = gspeed + (np.array(yaw) / 2)
            #test = np.array(yaw)
            #test = gspeed
            testm = np.mean(test)
            #print len(test)
            sdata.append((sid,testm))
        return np.array(sdata)

    tpmeans, freemeans, controlmeans = map(get_means, conditions)
    pooled = np.vstack((freemeans, controlmeans))
    
    t, p = scipy.stats.ttest_rel(tpmeans[:,1], freemeans[:,1])
    print 'tp vs free t: ' + str(t) + ' p: ' + str(p)
    t, p = scipy.stats.ttest_1samp(tpmeans[:,1], 0)
    print 'tp vs 0 t: ' + str(t) + ' p: ' + str(p)
    t, p = scipy.stats.ttest_1samp(freemeans[:,1], 0)
    print 'free vs 0 t: ' + str(t) + ' p: ' + str(p)
    t, p = scipy.stats.ttest_1samp(controlmeans[:,1], 0)
    print 'control vs 0 t: ' + str(t) + ' p: ' + str(p)
    t, p = scipy.stats.ttest_ind(freemeans[:,1], controlmeans[:,1])
    print 'control vs free t: ' + str(t) + ' p: ' + str(p)
    t, p = scipy.stats.ttest_1samp(pooled[:,1], 0)
    print 'free+control vs 0 t: ' + str(t) + ' p: ' + str(p)

    n = len(pooled[:,1])
    d = scipy.stats.t.ppf(0.975, n-1) * np.std(pooled[:,1], ddof=1) / np.sqrt(n)
    #d = 2*(np.std(pooled[:,1]) / np.sqrt(len(pooled[:,1])))
    print "\npooled 95% CI ["+ str(np.mean(pooled[:,1])-d) +", "+ str(np.mean(pooled[:,1])+d) +"]"
    
    n = len(tpmeans[:,1]) 
    #d = 2*(np.std(tpmeans[:,1]) / np.sqrt(len(tpmeans[:,1])))
    d = scipy.stats.t.ppf(0.975, n-1) * np.std(tpmeans[:,1], ddof=1) / np.sqrt(n)
    print "TP 95% CI ["+ str(np.mean(tpmeans[:,1])-d) +", "+ str(np.mean(tpmeans[:,1])+d) +"]"
    
    n = len(freemeans[:,1])
    #d = 2*(np.std(freemeans[:,1]) / np.sqrt(len(freemeans[:,1])))
    d = scipy.stats.t.ppf(0.975, n-1) * np.std(freemeans[:,1], ddof=1) / np.sqrt(n)
    print "FREE 95% CI ["+ str(np.mean(freemeans[:,1])-d) +", "+ str(np.mean(freemeans[:,1])+d) +"]"
   
    n = len(controlmeans[:,1])
    d = scipy.stats.t.ppf(0.975, n-1) * np.std(controlmeans[:,1], ddof=1) / np.sqrt(n)
    #d = 2*(np.std(controlmeans[:,1]) / np.sqrt(len(controlmeans[:,1])))
    print "CONTROL 95% CI ["+ str(np.mean(controlmeans[:,1])-d) +", "+ str(np.mean(controlmeans[:,1])+d) +"]"

    def welch_sthwaite(n1, n2, varx, vary):
        num = ( (varx / n1) + (vary / n2) ) **2
        denom = ( ( (varx/n1)**2 ) / (n1-1) ) + ( ( (vary/n2)**2 ) / (n2-1) )
        return num / denom

    n = len(controlmeans[:,1])
    m = len(freemeans[:,1])
    varx = np.var(controlmeans[:,1], ddof=1)
    vary = np.var(freemeans[:,1], ddof=1)
    delta_xy = np.mean(controlmeans[:,1]) - np.mean(freemeans[:,1])

    # control-free CI, welch version
    r = welch_sthwaite(n,m,varx,vary)
    d = scipy.stats.t.ppf(0.975, r) * np.sqrt((varx/n) + (vary/m))
    print "\nCONTROL-FREE CI, eq.var not assumed (welch)"
    print delta_xy-d, delta_xy+d
    
    # equal variaces assumed
    s_pooled = np.sqrt( ((n-1)*varx + (m-1)*vary) / (n+m-2) )
    d = scipy.stats.t.ppf(0.975, (n+m-2)) * s_pooled * np.sqrt( (1.0/n) + (1.0/m) )
    print "\nCONTROL-FREE CI, eq.var assumed"
    print delta_xy-d, delta_xy+d    
    
    #for x in freemeans[:,1]:
    #    print x

    
    


    """
    plt.figure()
    plt.hist(freemeans[:,1], bins=10)
    plt.show()
    plt.close()
    """

if __name__ == '__main__':
    #run_em()
    #plot_longdiff_lie()
    #plot_curve_map()
    #plot_gaze_vs_tp()
    #analyze_cornering_yaw_rate()
    #plot_delta_modes()
    #plot_aoi_catch()
    #plot_gaze_vs_tp_dist()
    #plot_gaze_delta_dist()
    #test_segmented_regression()
    #pursuit_location_vs_slope()
    #pursuit_endpoint_distributions()
    #gaze_location_density()
    #gaze_delta_density()
    #gaze_delta_density_plot_everything()
    #gaze_delta_density_free_vs_tp()
    #angle_density()
    #horiz_delta_vs_half_yaw()
    #plot_segmentations()
    #confidence_region_plot(c3_delta_modes)
    #tp_dist_vs_pursuit_angle()
    conditions_ttest()

