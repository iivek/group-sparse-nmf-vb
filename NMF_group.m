function [ tM, vM, indicators, aT, bT, indicatorShapes, indicatorScales, sigT, sigV, stats ] = NMF_group(...
    x, labeling, tInit, vInit,...
    aT, bT, indicatorShapes, indicatorScales,...
    burnin, update_t_hyperparams, update_lambda_hyperparams, nrPasses, monitorEvery)

% update_T_hyperparams (boll flag) ... optimize hyperparameters.
%   only one selection - a single pair of hyperparameters shared by all elements of T

% Algorithm does not attempt to learn labels.
%
% Algorithm may not be suitable for multilabel-driven decompositions. An
% example - if a factor is significant at labels "1", "2" and "5",
% contribution of this factor to a sample having label "1" is expected to
% be 3 times (very roughly) smaller than a sample being labeled as "1" and
% "2" and "5"

height = size(x,1);
width = size(x,2);
numFactors = size(bT, 2);

% % missing values
% M = full(isnan(x));
% numels = sum(sum(M));
% x(M) = zeros(numels,1);
% M = ~M;
% clear numels

% reshape labeling
labels = unique(labeling(~isnan(labeling)));
nrClasses = numel(labels);
labeling_ = zeros(1,width,nrClasses);
for l = 1:numel(labels)
    labeling_(1,:,l) = (labeling==labels(l));
end
%labeling_ = reshape(labeling, [1 size(labeling)]);

% now what we have is a logical variable labeling_ which for each sample
% indicates the labels, for all label layers. breakPoints indicates
% beginning and ending indices of labeling_ which belong to the same layer

%tInit = gamrnd(aT, bT);
%vInit = gamrnd(aV, bV);
%numFactors = size(vInit,1);
tM = zeros(height,numFactors,2);   % 2 natural parameters per variable
tM(:,:,1) = tInit;
tM(:,:,2) = log(tM(:,:,1) + 1e-12); % expectations of the reciprocals
vM = zeros(numFactors,width,2);
vM(:,:,1) = vInit;
vM(:,:,2) = log( vM(:,:,1) + 1e-12);
sigT = tM(:,:,1);
sigV = vM(:,:,1);

nrClasses = size(labeling_, 3);
%% adding a dimension to indicatorShapes and indicatorScales
indicatorShapes = reshape(indicatorShapes, [numFactors, 1, nrClasses]);
indicatorScales = reshape(indicatorScales, [numFactors, 1, nrClasses]);

% lambdas, the scale multipliers, continuous mixture of
% exponentials
%indicators(:,:,:,1) = gamrnd(indicatorShapes, indicatorScales);    % initial
indicators(:,:,:,1) = indicatorShapes.*indicatorScales;    % initial
indicators(:,:,:,2) = log(indicators(:,:,:,1)+1e-12);

entries = floor(nrPasses./monitorEvery);
stats.timestamps = zeros(entries,1);
stats.bounds = zeros(entries,1);
stats.sirs = zeros(entries,1);
stats.corrIndex = zeros(entries,1);
stats.sparsity = zeros(entries,1);
nextentry = 1;

%%
%%  Variational updates begin
%
% ordering will be natural per indices;
%
gammalnx = sparse(height,width);
gammalnx(find(x)) = gammaln(x(find(x)) + 1);
oldbound = -Inf;
%samples = find(M_labeling);
%initial stuff, so the bound calculation doesn't break. redundant in first
%pass.
alphaLambda = repmat(sum(labeling_(1,:,:),2), [numFactors 1 1])+indicatorShapes;
betaLambda = reshape( 1./(vM(:,:,1)*squeeze(labeling_(1,:,:))+1./squeeze(indicatorScales)), size(indicators(:,:,:,1)));
%initial stuff ends

for pass = 1:nrPasses
    pass
    
    temp = x./(exp( tM(:,:,2) )*exp( vM(:,:,2) ) ) ;
    sigT = exp( tM(:,:,2) ).*(temp*exp( vM(:,:,2) )');
    sigV = exp( vM(:,:,2) ).*(exp( tM(:,:,2) )'*temp);
    
    %% Updating the left matrix tM.
    alphaT = aT + sigT;
    % betaT = 1./( 1./bT + M*vM(:,:,1)' );
    betaT = 1./( 1./bT + repmat(sum(vM(:,:,1),2)',height,1) );
    tM(:,:,1) = alphaT.*betaT;
    
    %% Updating the right matrix vM
    temp = sum(repmat(labeling_, [numFactors 1 1]).*repmat(indicators(:,:,:,1),[1 width 1]),3);
    %temp = squeeze(indicators(:,1,:,1))*squeeze(labeling_(1,:,:))';   the
    %same as the above expression
    
    alphaV = sigV+1;
    %betaV  = 1./(temp + tM(:,:,1)'*M);
    betaV  = 1./(temp + repmat(sum(tM(:,:,1),1)', [1 width]));
    vM(:,:,1) = alphaV.*betaV;
    
    %% Updating the right matrix vM
    %     temp = sum(repmat(labeling_(:,~M_labeling,:), [numFactors 1 1]).*repmat(indicators(:,:,:,1),[1 nrHavingLabels 1]),3);
    %     alphaV1 = sigV(:,~M_labeling)+1;
    %     betaV1  = 1./(temp + tM(:,:,1)'*M(:,~M_labeling));
    %     vM(:,~M_labeling,1) = alphaV1.*betaV1;
    %     if(~(nnz(M_labeling)==0))
    %         alphaV2 = sigV(:,M_labeling)+1;
    %         betaV2  = 1./(1./bV + tM(:,:,1)'*M(:,M_labeling));
    %         vM(:,M_labeling,1) = alphaV2.*betaV2;
    %     end
    
    if(mod(pass,monitorEvery)==0 | pass == nrPasses)
        lT = exp(tM(:,:,2));
        lV = exp(vM(:,:,2));
        lTlV = lT*lV;
        %   temp = sum(repmat(labeling_, [numFactors 1 1]).*repmat(indicators(:,:,:,1),[1 width 1]),3);
        
        %% calculate bound here
        %bound = -sum(sum( M.*(tM(:,:,1)*vM(:,:,1) ) +  gammalnx )) ...
        bound = -sum(sum( tM(:,:,1)*vM(:,:,1) +  gammalnx )) ...
            + sum(sum( -x.*( ((lT.*tM(:,:,2))*lV + lT*(lV.*vM(:,:,2)))./(lTlV) -  log(lTlV) )  )) ...
            + sum(sum(-1./bT.*tM(:,:,1) - gammaln(aT) - aT.*log(bT)  )) ...
            + sum(sum( alphaT.*(log(betaT) + 1) + gammaln(alphaT) )) ...
            + sum(sum( -temp.*vM(:,:,1) + log(temp) ))...
            + sum(sum( alphaV.*(log(betaV) + 1) + gammaln(alphaV) )) ...
            + sum(sum(squeeze(-1./indicatorScales.*indicators(:,:,:,1) + (indicatorShapes-1).*indicators(:,:,:,2) ...
            -indicatorShapes.*log(indicatorScales)  - gammaln(indicatorShapes) ) ))...
            + sum(sum( (1-alphaLambda).*psi(alphaLambda) + log(betaLambda) + alphaLambda + gammaln(alphaLambda) ));
        
        diff = bound-oldbound;
        oldbound = bound;
        stats.bounds(nextentry) = bound;
        if diff <0
            diff
        end
        
        nextentry = nextentry + 1;
    end
    
    %% Updating what's left of tM and vM
    tM(:,:,2) = psi(alphaT)+log( betaT );
    vM(:,:,2) = psi(alphaV)+log( betaV );
    %     vM(:,~M_labeling,2) = psi(alphaV1)+log( betaV1 );
    %     if(~(nnz(M_labeling)==0))
    %         vM(:,M_labeling,2)  = psi(alphaV2)+log( betaV2 );
    %     end
    
    if pass>burnin
        %% Updating indicators - mixture weights
        alphaLambda = repmat(sum(labeling_(1,:,:),2), [numFactors 1 1])+indicatorShapes;
        betaLambda = reshape( 1./(vM(:,:,1)*squeeze(labeling_(1,:,:))+1./squeeze(indicatorScales)), size(indicators(:,:,:,1)));
        indicators(:,:,:,1) = alphaLambda.*betaLambda;
        indicators(:,:,:,2) = psi(alphaLambda)+log( betaLambda );
        
        %% Updating indicator variables only on variables which have labels
        %     alphaV = repmat(sum(labeling_(1,~M_labeling,:),2), [numFactors 1 1])+indicatorShapes;
        %     betaV = reshape( 1./(vM(:,~M_labeling,1)*squeeze(labeling_(1,~M_labeling,:))+1./indicatorScales), size(indicators(:,:,:,1)));
        %     indicators(:,:,:,1) = alphaV.*betaV;
        %     indicators(:,:,:,2) = psi(alphaV)+log( betaV );
    end
    %% TODO: refine the choices on type of updates (shared by rows/cols or
    %% shared by all variables)
    if( update_t_hyperparams )
        temp = invpsi( (sum(sum(tM(:,:,2)))-sum(sum(log(bT))))./numel(aT) );
        if temp>1
            % do not update if prior on T would become sparse because this
            % would inhibit the sparsity of V
            aT = repmat( temp , size(aT) );
        end
        bT = repmat( sum(sum( tM(:,:,1))./sum(aT(:)) ), size(bT) );
    end
    if( update_lambda_hyperparams )
        %        indicatorShapes = repmat( invpsi( (sum(sum(squeeze(indicators(:,:,:,2))))-sum(sum(log(squeeze(indicatorScales)))))./numel(indicatorShapes) ), size(indicatorShapes) );
        indicatorScales = repmat( sum(sum(squeeze(indicators(:,:,:,1))))./sum(indicatorShapes(:)), size(indicatorScales) );
        %indicatorShapes(1)
    end
    
    %figure(1); colormap hot; imagesc( 1./squeeze(indicators(:,:,:,1)) );
    %drawnow
    
end