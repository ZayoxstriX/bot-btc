_P='LOT_SIZE'
_O='filterType'
_N='filters'
_M='symbol'
_L='symbols'
_K='BTCUSDT'
_J='chikou_span'
_I='senkou_span_b'
_H='senkou_span_a'
_G='entryPrice'
_F='kijun_sen'
_E='tenkan_sen'
_D='valid'
_C='positionAmt'
_B='SHORT'
_A='LONG'
import numpy as np,pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from colorama import Fore,Style,init
import time,requests
SIDE_BUY='BUY'
SIDE_SELL='SELL'
ORDER_TYPE_MARKET='MARKET'
ORDER_TYPE_STOP_MARKET='STOP_MARKET'
ORDER_TYPE_LIMIT='LIMIT'
init(autoreset=True)
class TradingBot:
	def __init__(A,api_key,api_secret,leverage=50,symbol=_K):A.client=Client(api_key,api_secret);A.symbol=symbol;A.leverage=leverage;A.btc_amount=.0001;A.model=None;A.last_training_time=time.time();A.training_interval=1;A.setup_futures();A.suivi_en_temps_reel=True;A.seuil_risque=.02;A.alarmes=[]
	def suivre_en_temps_reel(A):
		while A.suivi_en_temps_reel:
			try:B=A.get_historical_data(Client.KLINE_INTERVAL_1MINUTE,100);C=B[:,4];A.verifier_conditions(C);time.sleep(10)
			except Exception as D:A.print_colored(f"Erreur lors du suivi en temps réel : {D}",Fore.RED)
	def verifier_conditions(A,close_prices):
		B=close_prices;D=A.compute_atr(B,14);C=A.get_btc_price()
		if C<B[-1]-D*1.5:A.open_position(_A)
		elif C>B[-1]+D*1.5:A.open_position(_B)
		A.verifier_alarmes(C)
	def verifier_alarmes(A,current_price):
		B=current_price;D=A.get_btc_price();C=abs((D-B)/B)
		if C>A.seuil_risque:A.envoyer_alarme(f"Alerte : Variation de {C*100:.2f}% détectée !")
	def envoyer_alarme(A,message):B=message;A.alarmes.append(B);A.print_colored(f"ALARME : {B}",Fore.RED)
	def gestion_automatique_positions(A):
		D=A.check_open_positions()
		for B in D:
			C=A.calculate_net_profit(float(B[_G]),A.get_btc_price(),float(B[_C]),A.leverage)
			if C<0:A.close_position(_A if float(B[_C])>0 else _B);A.envoyer_alarme(f"Fermeture automatique d'une position avec perte : {C} USDT")
	def print_colored(A,message,color=Fore.WHITE):print(f"{color}{message}{Style.RESET_ALL}")
	def setup_futures(A):
		try:A.client.futures_change_leverage(symbol=A.symbol,leverage=A.leverage);A.print_colored(f"Leverage set to {A.leverage} for {A.symbol}",Fore.GREEN)
		except BinanceAPIException as B:A.print_colored(f"Erreur lors de la configuration du levier : {B}",Fore.RED)
	def get_btc_price(A):
		try:B=A.client.get_symbol_ticker(symbol=A.symbol);return float(B['price'])
		except BinanceAPIException as C:A.print_colored(f"Erreur lors de la récupération du prix : {C}",Fore.RED);return 0
	def get_historical_data(A,interval,lookback):
		try:B=A.client.get_klines(symbol=A.symbol,interval=interval,limit=lookback);C=np.array([list(map(float,A[:5]))for A in B]);return C
		except BinanceAPIException as D:A.print_colored(f"Erreur lors de la récupération des données historiques : {D}",Fore.RED);return[]
	def generate_features(D,data):
		B=data[:,4];F=np.diff(B)/B[:-1];H=np.convolve(B,np.ones(20)/20,mode=_D);I=np.convolve(B,np.ones(50)/50,mode=_D);E=np.zeros_like(B);J=2/(10+1);E[0]=B[0]
		for G in range(1,len(B)):E[G]=J*B[G]+(1-J)*E[G-1]
		K=D.compute_rsi(B,14);L,M=D.compute_macd(B);N,O,P=D.compute_bollinger_bands(B);Q=D.compute_fibonacci_retracement(B);C=D.compute_ichimoku(data);A=min(len(F),len(H),len(I),len(E),len(K),len(L),len(M),len(N),len(O),len(P),len(Q),len(C[_E]));R=np.column_stack([F[-A:],H[-A:],I[-A:],E[-A:],K[-A:],L[-A:],M[-A:],N[-A:],O[-A:],P[-A:],Q[-A:],C[_E].values[-A:],C[_F].values[-A:],C[_H].values[-A:],C[_I].values[-A:],C[_J].values[-A:]]);S=(F[-A:]>0).astype(int);return R,S
	def compute_rsi(I,series,window):A=window;B=np.diff(series);D=np.where(B>0,B,0);E=np.where(B<0,-B,0);F=np.convolve(D,np.ones(A)/A,mode=_D);G=np.convolve(E,np.ones(A)/A,mode=_D);H=F/G;C=100-100/(1+H);C=np.concatenate([np.full(A-1,np.nan),C]);return C
	def calculate_profit(B,position):
		C=position;D=float(C[_G]);E=float(B.get_current_price());A=float(C[_C])
		if A<0:F=(D-E)*abs(A)
		else:F=(E-D)*A
		G=B.calculate_transaction_fees(A);H=F-G;return H
	def calculate_transaction_fees(B,amount):A=.001;return amount*A
	def calculate_net_profit(H,entry_price,current_price,position_size,leverage,fees_percentage=.0004):
		E=leverage;D=current_price;C=entry_price;A=position_size
		if A>0:B=(D-C)*A*E
		else:B=(C-D)*abs(A)*E
		F=abs(B)*fees_percentage;G=B-F;return G
	def compute_macd(I,series):
		A=series;C=np.zeros_like(A);D=np.zeros_like(A);G=2/(12+1);H=2/(26+1);C[0]=A[0];D[0]=A[0]
		for B in range(1,len(A)):C[B]=G*A[B]+(1-G)*C[B-1];D[B]=H*A[B]+(1-H)*D[B-1]
		F=C-D;E=np.convolve(F,np.ones(9)/9,mode=_D);E=np.concatenate([np.full(len(F)-len(E),np.nan),E]);return F,E
	def compute_bollinger_bands(I,series,window=20,num_std_dev=2):F=num_std_dev;E=series;A=window;E=pd.Series(E);B=E.rolling(window=A).mean();G=E.rolling(window=A).std();C=B+G*F;D=B-G*F;B=B.to_numpy();C=C.to_numpy();D=D.to_numpy();H=np.concatenate([np.full(A-1,np.nan),B[A-1:]]);C=np.concatenate([np.full(A-1,np.nan),C[A-1:]]);D=np.concatenate([np.full(A-1,np.nan),D[A-1:]]);return H,C,D
	def compute_fibonacci_retracement(F,series):B=series;A=np.max(B);E=np.min(B);C=A-E;D=[A-.236*C,A-.382*C,A-.618*C];return np.concatenate([np.full(len(B)-len(D),np.nan),D])
	def compute_atr(A,close_prices,period):B=close_prices;C=np.array(A.get_high_prices());D=np.array(A.get_low_prices());E=np.maximum(C-D,np.maximum(np.abs(C-np.roll(B,1)),np.abs(D-np.roll(B,1))));F=np.mean(E[-period:]);return F
	def get_high_prices(A):
		B=A.get_historical_data(Client.KLINE_INTERVAL_1MINUTE,100)
		if not B.size:A.print_colored('Erreur : Impossible de récupérer les données historiques pour les prix les plus hauts.',Fore.RED);return[]
		return B[:,2]
	def get_low_prices(A):
		B=A.get_historical_data(Client.KLINE_INTERVAL_1MINUTE,100)
		if not B.size:A.print_colored('Erreur : Impossible de récupérer les données historiques pour les prix les plus bas.',Fore.RED);return[]
		return B[:,3]
	def compute_ichimoku(E,data):D='close';C='low';B='high';A=pd.DataFrame(data,columns=['timestamp','open',B,C,D]);A[_E]=(A[B].rolling(window=9).max()+A[C].rolling(window=9).min())/2;A[_F]=(A[B].rolling(window=26).max()+A[C].rolling(window=26).min())/2;A[_H]=((A[_E]+A[_F])/2).shift(26);A[_I]=((A[B].rolling(window=52).max()+A[C].rolling(window=52).min())/2).shift(26);A[_J]=A[D].shift(-26);return A[[_E,_F,_H,_I,_J]]
	def ai_predictor(B,data):C,D=B.generate_features(data);E=TimeSeriesSplit(n_splits=5);A=RandomForestClassifier(n_estimators=100);F=cross_val_score(A,C,D,cv=E,scoring='accuracy');G=np.mean(F);B.print_colored(f"Précision moyenne du modèle (cross-validation) : {G:.2f}",Fore.GREEN);A.fit(C,D);return A
	def open_position(A,position_type,atr_multiplier=1.5,min_profit=3):
		G=min_profit;E=atr_multiplier;B=position_type;Q=A.get_precision();K=A.get_minimum_position_size();C=A.get_btc_price();A.close_conflicting_positions(B);H=A.get_historical_data(Client.KLINE_INTERVAL_1MINUTE,100)
		if not H.size:A.print_colored("Erreur : Impossible de récupérer les données historiques pour l'ATR.",Fore.RED);return
		L=H[:,4];F=A.compute_atr(L,14);M=C-F*E if B==_A else C+F*E;I=C+F*E if B==_A else C-F*E;D=max(.002,K);N=A.get_available_margin();O=D*C/A.leverage
		if N<O:A.print_colored('Erreur : Marge insuffisante pour ouvrir la position',Fore.RED);return
		J=abs(I-C)*D*A.leverage
		if J<G:A.print_colored(f"Profit attendu ({J:.2f} USDT) inférieur au seuil minimum ({G} USDT).",Fore.YELLOW);return
		try:R=A.client.futures_create_order(symbol=A.symbol,side=SIDE_BUY if B==_A else SIDE_SELL,type=ORDER_TYPE_MARKET,quantity=D,leverage=A.leverage);A.print_colored(f"Position {B} ouverte avec {D} BTC",Fore.GREEN);A.client.futures_create_order(symbol=A.symbol,side=SIDE_SELL if B==_A else SIDE_BUY,type=ORDER_TYPE_STOP_MARKET,stopPrice=M,quantity=D,leverage=A.leverage);A.client.futures_create_order(symbol=A.symbol,side=SIDE_SELL if B==_A else SIDE_BUY,type=ORDER_TYPE_LIMIT,price=I,quantity=D,leverage=A.leverage);A.print_colored(f"Stop-Loss et Take-Profit placés pour la position {B}",Fore.GREEN)
		except BinanceAPIException as P:A.print_colored(f"Erreur lors de l'ouverture de la position {B}: {P}",Fore.RED)
	def close_conflicting_positions(A,position_type):
		B=position_type;D=A.check_open_positions()
		for E in D:
			C=float(E[_C])
			if B==_A and C<0 or B==_B and C>0:
				F=SIDE_BUY if B==_B else SIDE_SELL
				try:A.client.futures_create_order(symbol=A.symbol,side=F,type=ORDER_TYPE_MARKET,quantity=abs(C),leverage=A.leverage);A.print_colored(f"Position conflictuelle fermée pour {B}",Fore.YELLOW);time.sleep(10)
				except BinanceAPIException as G:A.print_colored(f"Erreur lors de la fermeture de la position conflictuelle : {G}",Fore.RED)
	def close_position(A,position_type):
		B=position_type
		try:
			F=A.check_open_positions()
			if not F:A.print_colored('Aucune position ouverte à fermer',Fore.YELLOW);return
			for G in F:
				D=float(G[_C])
				if D>0:E=_A
				elif D<0:E=_B
				else:continue
				if B==_A and E==_A or B==_B and E==_B:
					H=SIDE_BUY if E==_B else SIDE_SELL
					try:
						I=A.get_available_margin();J=abs(D)*A.get_btc_price()/A.leverage
						if I<J:A.print_colored(f"Marge insuffisante pour fermer la position {B}.",Fore.RED);continue
						L=A.client.futures_create_order(symbol=A.symbol,side=H,type=ORDER_TYPE_MARKET,quantity=abs(D),leverage=A.leverage);A.print_colored(f"Position {B} fermée avec succès",Fore.GREEN);K=A.calculate_profit(G);A.print_colored(f"Profit net de la position {B} : {K:.2f} USDT",Fore.GREEN)
					except BinanceAPIException as C:
						if C.code==-2019:A.print_colored('Erreur : Marge insuffisante pour fermer la position.',Fore.RED)
						else:A.print_colored(f"Erreur lors de la fermeture de la position {B} : {str(C)}",Fore.RED)
					except Exception as C:A.print_colored(f"Erreur générale lors de la fermeture de la position {B} : {str(C)}",Fore.RED)
		except Exception as C:A.print_colored(f"Erreur générale lors de la fermeture de la position {B} : {str(C)}",Fore.RED)
	def ai_trading_bot(A):
		while True:
			A.print_colored('Récupération des données historiques...',Fore.CYAN);D=A.get_historical_data(Client.KLINE_INTERVAL_1MINUTE,100)
			if not D.size:A.print_colored('Erreur : Impossible de récupérer les données historiques.',Fore.RED);time.sleep(60);continue
			E=time.time()
			if E-A.last_training_time>A.training_interval:A.print_colored("Réentraînement du modèle d'IA...",Fore.CYAN);A.model=A.ai_predictor(D);A.last_training_time=E
			if A.model is not None:
				I,N=A.generate_features(D);F=I[-1].reshape(1,-1);C=A.model.predict(F);J=A.model.predict_proba(F)[0]
				if J.max()<.7:A.print_colored('Prédiction peu fiable, ignorer le signal',Fore.YELLOW);time.sleep(60);continue
				if C==1:A.open_position(_A)
				elif C==0:A.open_position(_B)
				K=A.check_open_positions()
				for G in K:
					B=float(G[_C]);L=float(G[_G]);M=A.get_btc_price();H=A.calculate_net_profit(L,M,B,A.leverage)
					if B>0 and C==0 or B<0 and C==1:
						if H>0:A.close_position(_A if B>0 else _B);A.print_colored(f"Position {B} fermée avec un profit net de {H:.2f} USDT",Fore.GREEN)
						else:A.print_colored(f"Position {B} non fermée. Profit net insuffisant.",Fore.YELLOW)
			time.sleep(60)
	def check_margin(A):B=A.client.futures_account();C=float(B['totalMarginBalance']);A.print_colored(f"Marge disponible: {C}",Fore.YELLOW)
	def check_open_positions(A):B=A.client.futures_position_information(symbol=A.symbol);return[A for A in B if float(A[_C])!=0]
	def get_precision(A):
		try:
			C=A.client.futures_exchange_info();D=C[_L]
			for B in D:
				if B[_M]==A.symbol:
					for filter in B[_N]:
						if filter[_O]==_P:E=int(-np.log10(float(filter['stepSize'])));return E
		except BinanceAPIException as F:A.print_colored(f"Erreur lors de la récupération des informations de précision : {F}",Fore.RED);return 0
	def get_available_margin(A):
		try:B=A.client.futures_account();return float(B['totalWalletBalance'])
		except BinanceAPIException as C:A.print_colored(f"Erreur lors de la récupération de la marge disponible : {C.message}",Fore.RED);return 0
	def get_minimum_position_size(A):
		try:
			C=A.client.futures_exchange_info();D=C[_L]
			for B in D:
				if B[_M]==A.symbol:
					for filter in B[_N]:
						if filter[_O]==_P:E=float(filter['minQty']);return E
		except BinanceAPIException as F:A.print_colored(f"Erreur lors de la récupération des informations de taille minimale : {F.message}",Fore.RED);return 0
if __name__=='__main__':api_key='dUE2KPUPaoZCKCsaGZzLqj14QkxdkXNIeaCkHETn9LyVsMnlVsSWQxsqwBQ4LxK4';api_secret='NT789q9T45zISfdzLOIvmvbGU2d0xrzxubWjVvUkT4FoiRBDR1HJQvA7XT53ydzd';bot=TradingBot(api_key,api_secret,leverage=20,symbol=_K);bot.ai_trading_bot()