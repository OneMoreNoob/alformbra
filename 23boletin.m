% matlab -nodesktop -nosplash -nosoftwareopengl < sprinkler.mat

semilla = 0; rng(semilla);
nMuestras = 100;
muestras = cell(N, nMuestras);
for i=1:nMuestras muestras(:,i) = sample_bnet(redB); end

redAPR        = mk_bnet(grafo, tallaNodos);
redAPR.CPD{C} = tabular_CPD(redAPR, C);
redAPR.CPD{R} = tabular_CPD(redAPR, R);
redAPR.CPD{S} = tabular_CPD(redAPR, S);
redAPR.CPD{W} = tabular_CPD(redAPR, W);

redAPR2=learn_params(redAPR, muestras);

% Se pueden mostrar las probabilidades estimadas como sigue:
TPCaux = cell(1,N);
for i=1:N s=struct(redAPR2.CPD{i}); TPCaux{i}=s.CPT; end

%%


% Aprendizaje con datos incompletos mediante EM

muestrasS = muestras;
semilla = 0; rng(semilla);
ocultas   = rand(N, nMuestras) > 0.5;
[I,J]     = find(ocultas);
for k=1:length(I) muestrasS{I(k), J(k)} = []; end

Preparamos una nueva red cuyas probabilidades van a ser re-estimadas por EM:
redEM = mk_bnet(grafo, tallaNodos, 'discrete', nodosDiscretos);
redEM.CPD{C} = tabular_CPD(redEM, C);
redEM.CPD{R} = tabular_CPD(redEM, R);
redEM.CPD{S} = tabular_CPD(redEM, S);
redEM.CPD{W} = tabular_CPD(redEM, W);
motorEM      = jtree_inf_engine(redEM);

maxIter = 100; eps = 1e-4;
semilla = 0; rng(semilla);
[redEM2, trazaLogVer] = learn_params_em(motorEM, muestrasS, maxIter, eps);
auxTPC = cell(1,N);
for i=1:N s=struct(redEM2.CPD{i}); auxTPC{i}=s.CPT; end
