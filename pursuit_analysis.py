import sys

import numpy as np
import tables
from constants import *
from memorize import memorize
from utils import get_merged_data, get_interp_naksu, pxx2heading, get_integrated, get_pursuit_fits, get_angled_range_data2, get_range_data, get_segment_data, filter_by_treatment, get_pooled
from tru.rec import foreach, rowstack, colstack, rec_degroup, groupby_multiple, groupby, groupby_i
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import itertools

from scipy.stats.kde import gaussian_kde
from statsmodels import api as sm

from matplotlib.backends.backend_pdf import PdfPages
from tru.se_scenecam import screencapture_to_angles
from tru.car import toyota_yaw_rate_in_degrees

def get_tp_referenced(data):
    refx = data['scenecam_x'] - data['naksu_x']
    refy = data['scenecam_y'] - data['naksu_y']
    rec = np.rec.fromarrays((refx, refy), names='tp_referenced_x, tp_referenced_y')
    data = colstack(data, rec)
    return data
    
def histogram_hdr(hist, bin_size, mass_limits):
    # Not very fast!
    ordered = np.sort(hist.flatten())[::-1]
    cum_dens = np.cumsum(ordered*bin_size)
    
    density_limits = np.searchsorted(cum_dens, mass_limits)
    return ordered[density_limits]

def get_pursuits_and_data_by_range(range):
    t1, t2 = range
    data = get_angled_range_data2(t1, t2)
    segs = get_pursuit_fits(t1, t2)
    
    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
    return groups, data, grp, segs

def fixation_frequencies(range):
    
    groups, data, grp, segs = get_pursuits_and_data_by_range(range)
    
    plt.figure()
    for sid, d in groupby(data, 'session_id'):
        seg = [segs[i] for i in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)
        seg = seg[(seg.t1 - seg.t0) > 0.1] # take out saccades
        laps = groups[groups['session_id'] == sid]
        laps = laps['lap']
        mean_pursuits = 1.0*len(seg)/len(laps)
        mean_vel = np.mean(d['c_vel'])
        
        plt.plot(mean_vel, mean_pursuits, '.k')
        
        print "%s: pursuits %i, mean_per_lap %f, mean_vel: %f" % (sid, len(seg), mean_pursuits, mean_vel)
    plt.show()
    
def pursuit_vs_yaw(range):
    groups, data, grp, segs = get_pursuits_and_data_by_range(range)
    pdf_out = PdfPages('/tmp/pursuit_vs_yaw.pdf')
    
    bin_size = 0.5
    
    means = []
    slopes = []
    noises = []
    for sid, d in groupby(data, 'session_id'):

        seg = [segs[i] for i in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)
        #seg = seg[(seg.t1 - seg.t0) > 0.2] # take out saccades
        seg = seg[seg.n > 12]
        
        # filter outliers from gaze landings
        values = np.vstack((seg.d0[:,0], seg.d0[:,1]))
        fkde = gaussian_kde(values)
        x, y = np.mgrid[np.min(values[0]):np.max(values[0]):bin_size, 
                        np.min(values[1]):np.max(values[1]):bin_size]
        grid = np.vstack((x.ravel(), y.ravel()))
        z = fkde(grid)
        reshaped = z.reshape(x.shape)
        medianhdr = histogram_hdr(z, bin_size**2, 0.8)
        seg = [s for s in seg if fkde((s.d0[0], s.d0[1])) > medianhdr]
        seg = np.hstack(seg).view(np.recarray)
        
        # mean yaw within each pursuit
        withinPursuitYaw = []
        for s in seg:
            slice = d[(d['ts'] >= s['t0']) &
                      (d['ts'] <= s['t1'])]
            ym = np.mean(toyota_yaw_rate_in_degrees(slice['c_yaw']))
            withinPursuitYaw.append(ym)
        withinPursuitYaw = np.array(withinPursuitYaw)
        
        dt = seg.t1 - seg.t0
        valid = dt > 0
        dt = dt[valid]
        gspeeds = -((seg.d1[:,0] - seg.d0[:,0]) / dt)
        noise = withinPursuitYaw / 2 - gspeeds
        
        plt.figure()
        # raw scatter
        fit = np.polyfit(withinPursuitYaw, gspeeds, 1)
        rng = np.linspace(10, 20, len(withinPursuitYaw))
        rline = np.polyval(fit, rng)
        model = np.poly1d([0.5, 0])
        plt.subplot(221)
        plt.plot(withinPursuitYaw, gspeeds, '.k', alpha=0.2)
        plt.plot(rng, rline, '-b')
        plt.plot([10,20], [np.polyval(model, 10), np.polyval(model, 20)], '-g')
        plt.ylim(-50,50)
        plt.xlim(5,25)
        plt.xlabel('mean yaw deg/s')
        plt.ylabel('gaze speed deg/s')
        plt.annotate('spearman r: %f' % scipy.stats.spearmanr(withinPursuitYaw, gspeeds)[0], (10,-10))
        plt.annotate('slope: %f' % fit[0], (10,-20))
        plt.title(sid)
        
        # noise kde + pdf
        plt.subplot(222)
        kde = gaussian_kde(noise)
        rng = np.linspace(np.min(noise), np.max(noise), len(noise))
        plt.plot(rng, kde(rng), '-b', label='kde')
        plt.plot(rng, scipy.stats.norm.pdf(rng, loc=np.mean(noise), scale=np.std(noise)), '-r', label='pdf')
        plt.hist(noise, bins=30, normed=True, color='green')
        plt.xlabel('mean yaw / 2 - gaze speed')
        plt.legend(loc='upper right')
        
        # pursuit landings scatter & density map
        plt.subplot(223)
        plt.contour(x,y,reshaped, [medianhdr])
        plt.plot(seg.d0[:,0], seg.d0[:,1], ',k')
        plt.xlim(-60,60)
        plt.ylim(-20,20)
        
        pdf_out.savefig()
        plt.close()
        
        print '%i \nspearman r: %f' % (sid, scipy.stats.spearmanr(withinPursuitYaw, gspeeds)[0])
        
        # how many fixations are from right to left (74-90 degrees)
        #validspeed = gspeeds < 0
        #print '%f of fixations from right to left' % (1.0 * len(gspeeds[validspeed]) / len(gspeeds))
    
        print 'mean of yaws / gaze speeds %f' % (np.mean(gspeeds / withinPursuitYaw))#, scipy.stats.mode(np.around(withinPursuitYaw / gspeeds, 1))[0])
        print 'mean of mean yaw / 2 - gaze speed: %f' % (np.mean(withinPursuitYaw / 2 - gspeeds))
        print 'noise shapiro-wilk w: %f p: %f' % scipy.stats.shapiro(noise)
        #print 'median hdr %f' % medianhdr
        
        #print 'mean of yaws: %f' % np.mean(withinPursuitYaw)
        #print 'mean of gspeeds: %f' % np.mean(gspeeds)
        
        means.append((np.mean(withinPursuitYaw), np.mean(gspeeds)))
        slopes.append(fit)
        noises.append(noise)

    pdf_out.close()
    return (means, slopes, noises)
    
def pursuit_yaw_means():
    
    pdf_out = PdfPages('/tmp/pursuit_vs_yaw_pooled.pdf')
    
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
    
    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
    
    all_dpoints = []
    xrange = [12,18]
    for sid, i in groupby_i(data, 'session_id'):
        d = data[i]
        print sid
        seg = [segs[k] for k in np.flatnonzero(groups.session_id == sid)]
        seg = np.hstack(seg).view(np.recarray)
        seg = seg[seg.n > 12]
        
        dpoints = []
        for s in seg:
            dt = s.t1 - s.t0
            if dt == 0: continue
            gspeed = -((s.d1[0]-s.d0[0]) / dt)
            slice = d[(d['ts'] >= s.t0) &
                      (d['ts'] <= s.t1)]
            ym = np.mean(toyota_yaw_rate_in_degrees(slice['c_yaw']))
            naksu_x = pxx2heading(slice['naksu_x'])
            naksu_mean = (naksu_x[1] - naksu_x[0]) / dt
            dpoints.append((ym, gspeed, naksu_mean))
        
        all_dpoints.append(dpoints)
        dpoints = np.array(dpoints)
        print scipy.stats.pearsonr(dpoints[:,0], dpoints[:,1])
        plt.figure()
        plt.title(sid)
        plt.hist(dpoints[:,0], bins=50)
        #plt.xlim(xrange)
        pdf_out.savefig()
        plt.close()
    
    all_dpoints = np.array(all_dpoints)
    means = np.array([np.median(x,0) for x in all_dpoints])
    ymeans = means[:,0]
    gmeans = means[:,1]
    nmeans = means[:,2]
    
    #from utils import fitLine
    #fitLine(ymeans,gmeans)
    
    fit = np.polyfit(ymeans, gmeans, 1)
    lsr = np.poly1d(fit)
    model = np.poly1d([0.5, 0])
    
    plt.figure()
    plt.plot(ymeans,gmeans,'.k')
    plt.plot(xrange, lsr(xrange), '-b')
    plt.plot(xrange, model(xrange), '-g')
    plt.xlabel('mean yaw rate (deg/s)')
    plt.ylabel('mean gaze speed (deg/s)')
    pdf_out.savefig()
    plt.close()
    
    nfit = np.polyfit(ymeans, nmeans, 1)
    nlsr = np.poly1d(nfit)
    plt.figure()
    plt.plot(ymeans,nmeans,'.k')
    plt.plot(xrange, nlsr(xrange), '-b')
    plt.xlabel('mean yaw rate (deg/s)')
    plt.ylabel('mean change in horizontal TP coordinate during pursuit (deg/s)')
    pdf_out.savefig()
    plt.close()
    
    pdf_out.close()
    
    print '\nspearman r: %f, p %f' % scipy.stats.spearmanr(ymeans, gmeans)
    print 'pearson r: %f, p %f' % scipy.stats.pearsonr(ymeans,gmeans)
    print 'least-squares fit: %f %f' % (fit[0], fit[1])
    print 'yaw/TP location least-squares fit: %f %f' % (nfit[0], nfit[1])
    #print 'slope shapiro-wilk w: %f p: %f' % scipy.stats.shapiro(coeffs)

def get_pooled_data(cis, is_control):
    arr_desc = [('sid', int), ('bend', int), ('yaw', float), ('speed', float)]
    table = np.array([], dtype=arr_desc)
    
    #cis = [1,2,3]
    for ci in cis:
        data = get_angled_range_data2(CORNERING[ci][0], CORNERING[ci][1])
        segs = get_pursuit_fits(CORNERING[ci][0], CORNERING[ci][1])
        
        data, segs = filter_by_treatment(data, segs, is_control)
        
        data.sort(order=['session_id', 'ts'])
        grp, segs = zip(*segs)
        groups = np.rec.fromrecords(grp, names='session_id,lap')

        for (sid, lap), d in groupby_multiple(data, ('session_id', 'lap')):
            #if (np.sum(d['g_direction_q'] < 0.2) > 0.25*(len(d))): continue
            seg = [segs[k] for k in np.flatnonzero((groups.session_id == sid) & 
                                                   (groups.lap == lap))]
            if (len(seg) == 0): continue
            seg = np.hstack(seg).view(np.recarray)
            seg = seg[seg.n > 12]
            
            lap_dpoints = []
            for s in seg:
                dt = s.t1 - s.t0
                if dt == 0: continue
                gspeed = -((s.d1[0]-s.d0[0]) / dt)
                slice = d[(d['ts'] >= s.t0) &
                          (d['ts'] <= s.t1)]
                ym = np.mean(toyota_yaw_rate_in_degrees(slice['c_yaw']))
                lap_dpoints.append((ym, gspeed))
            
            lap_dpoints = np.array(lap_dpoints)
            
            if (len(lap_dpoints > 0)):
                yaw = np.mean(lap_dpoints[:,0])
                speed = np.median(lap_dpoints[:,1])
                arr = np.array([(sid,ci,yaw,speed)], dtype=arr_desc)
                table = np.append(table, arr)
    return table

def pursuit_vs_yaw_all():
    ci = [1,3]
    from tru.ols_confidence_region import regression_ellipsoid
    from tru.studies.ramppi11.pursuit_analysis import get_pooled_data as r11_gpd
    r11 = r11_gpd(ci)
    r13free = get_pooled_data(ci, is_control=True)
    r13tp = get_pooled_data(ci, is_control=False)
    tables = (r11, r13free, r13tp)

    def collect(table):
        means = []
        for sid in np.unique(table['sid']):
            d = table[table['sid'] == sid]
            ymean = np.mean(d['yaw'])
            smean = np.mean(d['speed'])
            means.append((ymean, smean))
        return means

    def plot_slopes(subject_means, color, label):
        rlm_coeff = rlm_coeffs(subject_means)
        ellipser = regression_ellipsoid(subject_means[:,0], subject_means[:,1])
        t = np.linspace(0, np.pi*2, 100)
        ex, ey = ellipser(t)
        plt.plot(rlm_coeff.c[0], rlm_coeff.c[1], marker='o', color=color, label=label)
        plt.plot(ex, ey, color=color, linestyle='--', label='95% confidence region')
        plt.xlabel('Slope')
        plt.ylabel('Intercept')
        
    def plot_delta(subject_means, color, label):
        rlm_coeff = rlm_coeffs(subject_means)
        subject_means = np.array(subject_means)
        plt.plot(subject_means[:,0], subject_means[:,1], marker='.', linestyle='None', color=color)
        plt.plot(xrange, rlm_coeff(xrange), linestyle='-', color=color, label=label)
        plt.xlabel('Mean yaw rate (deg/s)')
        plt.ylabel('Mean gaze speed (deg/s)')

    def rlm_coeffs(subject_means):
        subject_means = np.array(subject_means)
        params = np.hstack([subject_means[:,0].reshape(-1,1),np.ones(len(subject_means[:,0])).reshape(-1,1)])
        rlm_model = sm.RLM(subject_means[:,1].reshape(-1,1), params, M=sm.robust.norms.HuberT())
        rlm_results = rlm_model.fit()
        rlm_coeff = np.poly1d(rlm_results.params)
        return rlm_coeff


    means = map(collect, tables)
    #means[-1] = means[-1][:-2]
    xrange=(8,18)
    colors = iter(['blue', 'green', 'red'])
    labels = iter(['r11', 'r13free', 'r13tp'])
    yaw_model = np.poly1d([0.5, 0])
    tp_model = np.poly1d([0,0])

    plt.figure()
    map(plot_delta, means, colors, labels)
    plt.plot(xrange, yaw_model(xrange), 'b--', label='Future path model')
    plt.plot(xrange, tp_model(xrange), 'r--', label='TP model')
    plt.ylim(None,12)
    #plt.plot(0.5, 0, marker='o', color='purple', label='Future path model')
    #plt.plot(0, 0, 'oy', label='Tangent point model')
    plt.legend(loc='upper left', numpoints=1)
    plt.show()
    plt.close()



def pursuit_yaw_new():
   
    ci = [1,3] 
    is_control = True
    txt = "free" if is_control else "tp"
    pdf_out = PdfPages('/tmp/pursuit_vs_yaw_pooled_per_subject_'+str(ci)+'_'+txt+'.pdf')
    table = get_pooled_data(ci, is_control)

    #invalid = (table['sid'] == 2011081107) & (table['speed'] < -5)
    #table = table[ ~invalid ]
    
    xrange = (0,20)
    
    sids = np.unique(table['sid'])
    subject_means = []
    lsr_coeffs = []
    for sid in sids:
        d = table[table['sid'] == sid]
        ymean = np.mean(d['yaw'])
        smean = np.mean(d['speed'])
        subject_means.append((ymean, smean))
        lsr_fit = np.polyfit(d['yaw'], d['speed'], 1)
        lsr = np.poly1d(lsr_fit)
        lsr_coeffs.append(lsr)
        #print sid, lsr.c, scipy.stats.pearsonr(d['yaw'], d['speed'])
        
        plt.figure()
        plt.title(sid)
        plt.xlim(xrange)
        plt.ylim(-20,20)
        plt.plot(d['yaw'], d['speed'], '.k')
        plt.plot(xrange, lsr(xrange),'-b')
        #plt.plot(xrange, model(xrange), '-g')
        plt.xlabel('mean yaw during fixation (deg)')
        plt.ylabel('gaze speed (deg/s)')
        pdf_out.savefig()
        plt.close()
        
    subject_means = np.array(subject_means)
    lsr_bs = np.array([x.c[0] for x in lsr_coeffs])
    lsr_as = np.array([x.c[1] for x in lsr_coeffs])
    
    fit_all = np.polyfit(table['yaw'], table['speed'], 1)
    lsr_all = np.poly1d(fit_all)
    print 'lsr all laps b=%f' % (lsr_all.c[0])
    
    #fit_subject = np.polyfit(subject_means[:,0], subject_means[:,1], 1)
    #lsr_subject = np.poly1d(fit_subject)
    params = np.hstack([subject_means[:,0].reshape(-1,1),np.ones(len(subject_means[:,0])).reshape(-1,1)])
    ols_model = sm.OLS(subject_means[:,1].reshape(-1,1), params)
    rlm_model = sm.RLM(subject_means[:,1].reshape(-1,1), params, M=sm.robust.norms.HuberT())
    yaw_model = np.poly1d([0.5, 0])
    ols_results = ols_model.fit()
    rlm_results = rlm_model.fit()
    ols_coeff = np.poly1d(ols_results.params)
    rlm_coeff = np.poly1d(rlm_results.params)
    print 'ols subjects b=%f' % (ols_coeff.c[0])
    print 'rlm subjects b=%f' % (rlm_coeff.c[0])
    print ols_results.summary()
    
    # binomial test
    n = len(lsr_bs)
    s = np.sum(lsr_bs > 0)
    p = 0.5
    p_binomial = np.math.factorial(n) / (np.math.factorial(s) * np.math.factorial(n-s)) * p**s * p**(n-s)
    print 'binomial p(%i) = %f' % (s, p_binomial)

    sy = np.std(subject_means[:,1])
    print 'standard deviation of gaze speed %f' % sy
    ss_residual = np.sum( (subject_means[:,1]-ols_coeff(subject_means[:,0]))**2 )
    sxy = np.sqrt( ss_residual / (len(subject_means[:,0]-2)) )
    print 'standard error of estimate %f' % sxy
    model_ss_residual = np.sum( (subject_means[:,1]-yaw_model(subject_means[:,0]))**2 )
    model_error = np.sqrt( model_ss_residual / (len(subject_means[:,0]-2)) )
    print 'error of estimate from model %f' % model_error
    
    # F stat
    n = len(subject_means[:,0])
    df1 = 2
    df2 = n-df1
    f = ((model_ss_residual-ss_residual) / df1) / (ss_residual / df2)
    p_f = 1-scipy.stats.f(df1,df2).cdf(f)
    print 'F=%f, p=%f' % (f, p_f)

    # confidence region stuff
    from tru.ols_confidence_region import regression_ellipsoid
    ellipser = regression_ellipsoid(subject_means[:,0], subject_means[:,1])
    t = np.linspace(0, np.pi*2, 100)
    ex, ey = ellipser(t)

    plt.figure()
    plt.plot(table['yaw'], table['speed'], '.k')
    plt.plot(xrange, lsr_all(xrange), '-b')
    pdf_out.savefig()
    plt.close()
    
    plt.figure()
    plt.plot(subject_means[:,0], subject_means[:,1], '.k')
    plt.plot(xrange, ols_coeff(xrange), '-g', label='Ordinary least squares fit')
    plt.plot(xrange, rlm_coeff(xrange), '-r', label='huber t')
    plt.plot(xrange, yaw_model(xrange), 'b--', label='Future path model')
    plt.xlabel('Mean yaw rate (deg/s)')
    plt.ylabel('Mean gaze speed (deg/s)')
    plt.legend(loc='lower right')
    pdf_out.savefig()
    plt.close()
    
    plt.figure()
    #plt.plot(lsr_bs, lsr_as, '.k')
    plt.plot(ols_coeff.c[0], ols_coeff.c[1], 'og', label='OLS fit')
    plt.plot(rlm_coeff.c[0], rlm_coeff.c[1], 'oy', label='huber t fit')
    plt.plot(0.5, 0, 'ob', label='Future path model')
    plt.plot(0, 0, 'or', label='Tangent point model')
    plt.plot(ex, ey, 'k--', label='95% confidence region')
    plt.xlabel('Slope')
    plt.ylabel('Intercept')
    plt.legend(loc='upper right', numpoints=1)
    pdf_out.savefig()
    plt.close()
    """
    model_errors = subject_means[:,1]-ols_coeff(subject_means[:,0]) 
    plt.figure()
    plt.hist(model_errors, bins=10)
    pdf_out.savefig()
    plt.close()
    """        
    pdf_out.close()

def tp_vel():
    
    def get_range_in_naksu(ldata, nsdata, ci):
        seg = ldata[ (ldata['dist'] >= CORNERING[ci][0]) & (ldata['dist'] <= CORNERING[ci][1]) ]
        nseg = nsdata[ (nsdata['se_ts'] >= np.nanmin(seg['se_frame_number'])) &
                       (nsdata['se_ts'] <= np.nanmax(seg['se_frame_number']))]
        nseg['x'], nseg['y'] = np.degrees(screencapture_to_angles(nseg['x'], nseg['y'], strict=False))
        return nseg
    
    def eucl_dist(seg):
        xd = np.abs(np.diff(seg['x']))
        yd = np.abs(np.diff(seg['y']))
        return np.sqrt(xd**2 + yd**2)
        
    cdata, naksu_data = get_integrated(DATA_FILE)
    sids = np.unique(naksu_data['session_id'])
    
    b1_h_delta = []
    b2_h_delta = []
    b3_h_delta = []
    for sid in sids:
        print sid
        sdata = cdata[ cdata['session_id'] == sid ]
        nsdata = naksu_data[ naksu_data['session_id'] == sid]
        laps = np.unique(sdata['lap'])
        for lap in laps:
            #print lap
            ldata = sdata[ sdata['lap'] == lap ]
            seg1 = get_range_in_naksu(ldata, nsdata, 1)
            seg2 = get_range_in_naksu(ldata, nsdata, 2)
            seg3 = get_range_in_naksu(ldata, nsdata, 3)
            b1_h_delta.extend(np.diff(seg1['x']))
            b2_h_delta.extend(np.diff(seg2['x']))
            b3_h_delta.extend(np.diff(seg3['x']))
            
            """dist1 = eucl_dist(seg1)
            dist2 = eucl_dist(seg2)
            dist3 = eucl_dist(seg3)
            
            plt.figure()
            plt.plot(dist1, '-r')
            plt.plot(dist2, '-g')
            plt.plot(dist3, '-b')
            plt.scatter(seg1['x'], seg1['y'], c='r')
            plt.scatter(seg2['x'], seg2['y'], c='g')
            plt.scatter(seg3['x'], seg3['y'], c='b')
            #plt.scatter(seg1['scenecam_x'], seg1['scenecam_y'])
            plt.show()"""
    
    b1_h_delta = np.array(b1_h_delta)
    b1_h_delta = b1_h_delta[np.isfinite(b1_h_delta)]*5
    b2_h_delta = np.array(b2_h_delta)
    b2_h_delta = b2_h_delta[np.isfinite(b2_h_delta)]*5
    b3_h_delta = np.array(b3_h_delta)
    b3_h_delta = b3_h_delta[np.isfinite(b3_h_delta)]*5
    
    pooled = np.hstack((b1_h_delta, b2_h_delta, b3_h_delta))
    percentiles = np.percentile(pooled, (1,5,10,25,50,75,90,95,99))
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = percentiles
    #plt.figure()
    #plt.hist(b1_h_delta, bins=100)
    #plt.show()
    
    print 'horiz. tp diff mean deg/s'
    print 'b1 %f, sd %f' % (np.mean(b1_h_delta), np.std(b1_h_delta))
    print 'b2 %f, sd %f' % (np.mean(b2_h_delta), np.std(b2_h_delta))
    print 'b3 %f, sd %f' % (np.mean(b3_h_delta), np.std(b3_h_delta))
    print 'pooled %f, sd %f' % (np.mean(pooled), np.std(pooled))
    print 'percentiles: \n1 %f \n5 %f \n10 %f \n25 %f \n50 %f \n75 %f \n90 %f \n95 %f \n99 %f' % (p1, p5, p10, p25, p50, p75, p90, p95, p99)

def plot_naksu_by_sid_and_bend():
    data = get_merged_data()
    ci = 3
    pdf_out = PdfPages('/tmp/naksu_by_sid_and_bend_'+str(ci)+'.pdf')
    
    for sid in SESSIONS:
        sdata = data[ data['session_id'] == sid]
        plt.figure()
        plt.title(sid)
        for lap in np.unique(sdata['lap']):
            ldata = sdata[ sdata['lap'] == lap ]
            segdata = get_segment_data(sid, CORNERING[ci], ldata)
            plt.plot(segdata['dist'], segdata['naksu_x'], '-g')
        pdf_out.savefig()
        plt.close()
    
    pdf_out.close()
        
def yaw_variation(is_control=True):
    
    ci = [1,3]
    #is_control = False
    
    data, segs = get_pooled(ci)
    data, segs = filter_by_treatment(data, segs, is_control)
    
    smeans = []    
    for sid, d in groupby(data, 'session_id'):
        yaw = toyota_yaw_rate_in_degrees(d['c_yaw'])
        mean = np.mean(yaw)
        std = np.std(yaw)
        print sid, mean, std
        smeans.append((sid, mean))
    #mean = np.mean(smeans)
    #std = np.std(smeans)
    #print 'mean: ' + str(mean) + ' std: ' + str(std)
    return smeans
        
def paired_ttest_yaw():
    control = np.array(yaw_variation(is_control=True))
    treatm = np.array(yaw_variation(is_control=False))
    t, p = scipy.stats.ttest_rel(control[:,1], treatm[:,1])
    print 'ttest rel: t('+str(len(control)-1)+')=' + str(t) + ', p=' + str(p)

def raw_pursuits_pdf(ci=[1], is_control=True):
    #ci = [3]
    #is_control = True
    txt = 'free' if is_control else 'tp'
    pdf_out = PdfPages('/tmp/raw_pursuits'+str(ci)+'_'+txt+'.pdf')
    
    data, segs = get_pooled(ci)
    data, segs = filter_by_treatment(data, segs, is_control)

    #print 'segs: ' + str(len(segs))
    
    grp, segs = zip(*segs)
    groups = np.rec.fromrecords(grp, names='session_id,lap')
    data = data[ data['g_direction_q'] > 0.4 ]

    for (sid, lap), d in groupby_multiple(data, ('session_id', 'lap')):
        seg = [segs[k] for k in np.flatnonzero((groups.session_id == sid) & 
                                               (groups.lap == lap))]
        #if (len(seg) == 0): continue
        seg = np.hstack(seg).view(np.recarray)
        seg = seg[seg.n > 12]

        plt.figure()
        
        ax1 = plt.subplot(211)
        plt.title(str(sid) + ', ' + str(lap))
        plt.plot(d['dist'], d['scenecam_x'], ',k')
        for s in seg:
            plt.plot([s.t0, s.t1], [s.d0[0], s.d1[0]], '-r')
        plt.ylim([-10,50])
        plt.ylabel('horiz.gaze (deg)')
        
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(d['dist'], d['scenecam_y'], ',k')
        for s in seg:
            plt.plot([s.t0, s.t1], [s.d0[1], s.d1[1]], '-r')
        plt.ylim([-15,15])
        plt.ylabel('vert.gaze (deg)')
        plt.xlabel('ts')
        pdf_out.savefig()
        plt.close()

    pdf_out.close()

def head_movement():
    data = get_merged_data()
    stats = []
    for sid in SESSIONS:
        sd = data[data['session_id'] == sid]
        gox = sd['g_origin_x']
        goy = sd['g_origin_y']
        m = (np.mean(gox), np.mean(goy))
        s = (np.std(gox), np.std(goy))
        stats.append((m,s))
        print sid
        print m[0], m[1]
        print s[0], s[1]
    stats = np.array(stats)
    m = stats[:,0]
    s = stats[:,1]
    print 'total:'
    print np.mean(m[:,0]), np.mean(m[:,1])
    print np.mean(s[:,0]), np.mean(s[:,1])

    """
    ci = [1,3]
    data, segs = get_pooled(ci)
    cdata, csegs = filter_by_treatment(data, segs, True)
    tdata, tsegs = filter_by_treatment(data, segs, False)
    head_c = (cdata['g_origin_x'], cdata['g_origin_y'])
    head_t = (tdata['g_origin_x'], tdata['g_origin_y'])
    print np.std(head_c), np.std(head_t)
    """

if __name__ == '__main__':
    
    #bendid = int(sys.argv[1])
    #range = CORNERING[bendid]
    
    """
    bendid = (1,2,3)
    means = []
    slopes = []
    noises =[]
    for i in bendid:
        range = CORNERING[i]
        m, s, n = pursuit_vs_yaw(range)
        means.extend(m)
        slopes.extend(s)
        noises.extend(n)
    """
    #print allstats
    #allstats = np.array(allstats).view(np.recarray).T
    #means, slopes = allstats

    #fixation_frequencies(range)
    #stats = pursuit_vs_yaw(range)
    #means, slopes = stats
    #pursuit_yaw_means()
    #pursuit_yaw_new()
    #pursuit_vs_yaw_all()
    #tp_vel()
    #plot_naksu_by_sid_and_bend()
    #paired_ttest_yaw()
    raw_pursuits_pdf()
    #head_movement()
        
    
    
    
    
    
