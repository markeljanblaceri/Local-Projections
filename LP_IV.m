clear all; close all; clc
warning off all

%% Load Data
filename = 'lpiv.xlsx'; % Excel file with time series data
sheet = 'Data'; % Worksheet containing the data
data = readtable(filename, 'Sheet', sheet); % Import data

% Store variables in a structured format
DATA = struct();
DATA = struct();
DATA.mps = data.MPS;                % Monetary policy shock
DATA.ip = 100*log(data.ip);         % Industrial production (log, scaled by 100)
DATA.inflation = data.inflation;    % Inflation
DATA.urate = data.urate;            % Unemployment rate
DATA.rate = data.ffr;               % Federal funds rate

%% Setup for LP-IV
H            = 24;               % Horizon
nlags        = 12;               % Lags
instrument   = DATA.mps;         % Instrument
Yvars   = {'ip','inflation','urate','rate'}; % Dependent variables
Ynames  = {'Industrial Production','Inflation','Unemployment Rate','Policy Rate'}; % Description of dependent variables
nvars   = length(Yvars); % Number ofdependent variables
nobs    = length(instrument); % Number of observations

T = (1:nobs)';          % Linear time trend
T2 = T.^2;              % Quadratic time trend

% Create matrix of endogenous variables
ENDO = [DATA.ip, DATA.inflation, DATA.urate, DATA.rate];

% Lag matrix of endogenous variables (used as controls)
Xlags = lagmatrix(ENDO, 1:nlags);

% Preallocate impulse response (IRF) and standard error (SE) matrices
LP_IRF = nan(H+1, nvars); % Stores IRFs
LP_SE  = nan(H+1, nvars); % Stores Newey-West SEs

%% LP-IV Loop

for h = 0:H
    for j = 1:nvars
        % Dependent variable at horizon h
        y = DATA.(Yvars{j});
        y_h = lagmatrix(y, -h);  % Lead of y: outcome h periods ahead
        
        % Remove missing values
        valid_idx = all(~isnan([y_h, Xlags, DATA.rate, instrument]), 2);
        y_h_clean = y_h(valid_idx);
        Xlags_clean = Xlags(valid_idx, :);
        rate_clean  = DATA.rate(valid_idx);
        instrument_clean = instrument(valid_idx);
        
        % Step 1: First stage regression
        % Regress endogenous rate on instrument + controls
        FS_X = [instrument_clean, Xlags_clean];
        fs_mdl = fitlm(FS_X, rate_clean, 'Intercept', true);
        rate_hat = fs_mdl.Fitted; % Predicted component of rate (instrumented)
        
        % Step 2: Second stage regression
        % Regress outcome at horizon h on instrumented rate + controls
        SS_X = [rate_hat, Xlags_clean]; 
        ss_mdl = fitlm(SS_X, y_h_clean, 'Intercept', true);
        b = ss_mdl.Coefficients.Estimate;
        
        % Compute HAC (Newey-West) standard errors
        if h == 0
            % At horizon 0: use OLS covariance
            HAC = ss_mdl.CoefficientCovariance;
        else
            bw = max(h,1);                 
            HAC = hac(SS_X, y_h_clean, 'Type','HAC','Bandwidth',bw);
        end
        
        % Store impulse response and its SE
        LP_IRF(h+1, j) = b(2);           % Coefficient on shock (first regressor)
        LP_SE(h+1, j)  = sqrt(HAC(2,2)); % Robust SE of shock coefficient
    end
end

disp('LP-IV IRFs computed successfully.');

%% IRFs Plot
ci68 = 1.0;    % ~68% confidence interval multiplier
ci95 = 1.96;   % ~95% confidence interval multiplier

% Labels for Y-axes
ylabelTexts    = {'Percent', 'Percentage Points', 'Percentage Points', 'Percentage Points'};

% Colors and line settings
Scolor         = [0 0 0.6];    % IRF line color (dark blue)    
bandFillColor1 = [0.5 0.7 1];  % Confidence band color
green     = [0 0.6 0];         % Zero line color

lwIRF  = 3;   % Line width of IRF
lwZero = 2;   % Line width of zero line

% Plot IRFs for each variable
for j = 1:nvars
    subplot(2,2,j) % Arrange plots in 2x2 grid
    
    hvec = 0:H;                
    irf  = LP_IRF(:, j);       
    se   = LP_SE(:, j);        

    % 95% confidence band
    upper95 = irf + ci95*se;
    lower95 = irf - ci95*se;
    fill([hvec, fliplr(hvec)], [upper95', fliplr(lower95')], ...
         [0.7 0.85 1], 'EdgeColor','none','FaceAlpha',0.4); hold on;

    % 68% confidence band
    upper68 = irf + ci68*se;
    lower68 = irf - ci68*se;
    fill([hvec, fliplr(hvec)], [upper68', fliplr(lower68')], ...
         [0.5 0.7 1], 'EdgeColor','none','FaceAlpha',0.6); hold on;

    % IRF line
    plot(hvec, irf, 'LineWidth', lwIRF, 'Color', Scolor); hold on;

    % Zero line
    plot(hvec, zeros(size(hvec)), '--', 'LineWidth', lwZero, 'Color', green);

    % Titles, labels, and formatting
    title(Ynames{j}, 'FontWeight','bold', 'FontName','Courier', 'FontSize',12);
    ylabel(ylabelTexts{j}, 'FontName','Courier','FontSize',12);
    xlabel('Horizon (quarters)','FontName','Courier','FontSize',12);
    xlim([0 H]); grid on; ax=gca;
    ax.GridLineStyle = '--'; ax.Box='off'; ax.TickDir='out';
end
