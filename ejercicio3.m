% matlab -nodesktop -nosplash -nosoftwareopengl < ej3Silvia.m

addpath('~/asigDSIC/ETSINF/apr/p2/BNT')
addpath(genpathKPM('~/asigDSIC/ETSINF/apr/p2/BNT'))

% cargar datos de aprendizaje
datApr       = load('./data/spam/tr.dat',       '-ascii');
etqApr       = load('./data/spam/trlabels.dat', '-ascii');
dataApr      = zscore(datApr);
etiqApr      = etqApr + 1;
[numVec dim] = size(dataApr)
numClas      = max(etiqApr)

grafo          = [ 0 1 1 ; 0 0 1 ; 0 0 0 ];
numNodos       = length(grafo)

% cargar datos de test
datTest  = load('./data/spam/ts.dat',       '-ascii');
etqTest  = load('./data/spam/tslabels.dat', '-ascii');
dataTest = zscore(datTest);
etiqTest = etqTest + 1;


nMuestrasTest = length(dataTest);

for numGaus = [1,2,5,10,20,50,100]
	numGaus
	
	% clasificador basado en mixtura de gaussianas 
	tallaNodos     = [numClas numGaus dim];
	nodosDiscretos = [1 2];
	redB           = mk_bnet(grafo, tallaNodos, 'discrete', nodosDiscretos);
	redB.CPD{1}    = tabular_CPD(redB, 1);
	redB.CPD{2}    = tabular_CPD(redB, 2);
	redB.CPD{3}    = gaussian_CPD(redB, 3, 'cov_type', 'diag');

	% poblar vector de entrenamiento
	datosApr             = cell(numNodos, numVec);
	datosApr(numNodos,:) = num2cell(dataApr', 1);
	datosApr(1,:)        = num2cell(etiqApr', 1);

	% motor de aprendizaje
	motor                = jtree_inf_engine(redB);
	maxIter              = 16;
	[redB2, ll, motor2]  = learn_params_em(motor, datosApr, maxIter);

	
	err = 0;
	p = zeros(nMuestrasTest, numClas); 	%% Limpiamos p por si se ha usado antes
	evidencia = cell(numNodos,1); 			%% Un cell array vacio para las observaciones
	for i=1:nMuestrasTest
		evidencia{numNodos} = dataTest(i,:)';
		[motor3, ll] = enter_evidence(motor2, evidencia);
		m = marginal_nodes(motor3, 1);
		p(i,:) = m.T';

		if p(i,1) == 1
			if etiqTest(i,:) == 1
			else
			    err= err +1;
			end
		else
			if etiqTest(i,:) == 2
			else
			    err = err +1;
			end
		end

	end;
	err
	porc_error = err/nMuestrasTest
	confianza = 1.96*sqrt(porc_error*(1-porc_error)/nMuestrasTest) % ¿bien calculado?
end

