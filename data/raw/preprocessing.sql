/*
copied from Data.sas.
*/

libname liv 'G:\All\Daten\livevol';
libname liv2 'K:\Daten\livevol 2017';
libname ise 'E:\Grauer\Trade classification\ISE\Datasets';
libname ise2 'K:\Daten\livevol 2017\ISE_2017\Datasets';
libname wd 'K:\Abschlussarbeiten\Markus Bilz';


data wd.livevol_quotes_remove;
	set liv.livevol_clean_6th(where=(exchange=6)) liv.livevol_clean_2015_6th(where=(exchange=6));
	drop underlying_symbol trade_condition_id canceled_trade_condition_id underlying_bid underlying_ask 
	number_of_exchanges exchange exchange_id_1-exchange_id_16 bid_size_1-bid_size_16 ask_size_1-ask_size_16 id;
run;


proc sql;
	create table wd.matched_ise_quotes
	as select a.*,
		b.ask_1, b.ask_2, b.ask_3, b.ask_4, b.ask_5, b.ask_6, b.ask_7, b.ask_8, b.ask_9, b.ask_10,
		b.ask_11, b.ask_12, b.ask_13, b.ask_14, b.ask_15, b.ask_16,
		b.bid_1, b.bid_2, b.bid_3, b.bid_4, b.bid_5, b.bid_6, b.bid_7, b.bid_8, b.bid_9, b.bid_10,
		b.bid_11, b.bid_12, b.bid_13, b.bid_14, b.bid_15, b.bid_16
	from ise.livevol_ise_merged2 as a 
	left join wd.livevol_quotes_remove as b
	on a.quote_datetime=b.quote_datetime and a.order_id = b.order_id and a.sequence_number = b.sequence_number
;
quit;

proc sql;
	create table wd.matched_ise_quotes_2017
	as select a.*,
		b.ask_1, b.ask_2, b.ask_3, b.ask_4, b.ask_5, b.ask_6, b.ask_7, b.ask_8, b.ask_9, b.ask_10,
		b.ask_11, b.ask_12, b.ask_13, b.ask_14, b.ask_15, b.ask_16,
		b.bid_1, b.bid_2, b.bid_3, b.bid_4, b.bid_5, b.bid_6, b.bid_7, b.bid_8, b.bid_9, b.bid_10,
		b.bid_11, b.bid_12, b.bid_13, b.bid_14, b.bid_15, b.bid_16
	from ise2.livevol_ise_2017_merged2 as a 
	left join liv2.livevol_clean_2017 as b
	on a.quote_datetime=b.quote_datetime and a.order_id = b.order_id and a.sequence_number = b.sequence_number
;
quit;

/*
data wd.livevol_ise_2004;
set liv.livevol_clean_6th(drop= underlying_symbol sequence_number root trade_condition_id canceled_trade_condition_id 
		underlying_bid underlying_ask number_of_exchanges);
where exchange= 6;

	if exchange_id_1 = exchange then do; ask_ise = ask_1; bid_ise = bid_1; bid_size_ise = bid_size_1; ask_size_ise = ask_size_1; end; else 
	if exchange_id_2 = exchange then do; ask_ise = ask_2; bid_ise = bid_2; bid_size_ise = bid_size_2; ask_size_ise = ask_size_2; end; else 
	if exchange_id_3 = exchange then do; ask_ise = ask_3; bid_ise = bid_3; bid_size_ise = bid_size_3; ask_size_ise = ask_size_3; end; else 
	if exchange_id_4 = exchange then do; ask_ise = ask_4; bid_ise = bid_4; bid_size_ise = bid_size_4; ask_size_ise = ask_size_4; end; else 
	if exchange_id_5 = exchange then do; ask_ise = ask_5; bid_ise = bid_5; bid_size_ise = bid_size_5; ask_size_ise = ask_size_5; end; else 
	if exchange_id_6 = exchange then do; ask_ise = ask_6; bid_ise = bid_6; bid_size_ise = bid_size_6; ask_size_ise = ask_size_6; end; else 
	if exchange_id_7 = exchange then do; ask_ise = ask_7; bid_ise = bid_7; bid_size_ise = bid_size_7; ask_size_ise = ask_size_7; end; else 
	if exchange_id_8 = exchange then do; ask_ise = ask_8; bid_ise = bid_8; bid_size_ise = bid_size_8; ask_size_ise = ask_size_8; end; else 
	if exchange_id_9 = exchange then do; ask_ise = ask_9; bid_ise = bid_9; bid_size_ise = bid_size_9; ask_size_ise = ask_size_9; end; else 
	if exchange_id_10 = exchange then do; ask_ise = ask_10; bid_ise = bid_10; bid_size_ise = bid_size_10; ask_size_ise = ask_size_10; end; else 
	if exchange_id_11 = exchange then do; ask_ise = ask_11; bid_ise = bid_11; bid_size_ise = bid_size_11; ask_size_ise = ask_size_11; end; else 
	if exchange_id_12 = exchange then do; ask_ise = ask_12; bid_ise = bid_12; bid_size_ise = bid_size_12; ask_size_ise = ask_size_12; end; else 
	if exchange_id_13 = exchange then do; ask_ise = ask_13; bid_ise = bid_13; bid_size_ise = bid_size_13; ask_size_ise = ask_size_13; end; else 
	if exchange_id_14 = exchange then do; ask_ise = ask_14; bid_ise = bid_14; bid_size_ise = bid_size_14; ask_size_ise = ask_size_14; end; else 
	if exchange_id_15 = exchange then do; ask_ise = ask_15; bid_ise = bid_15; bid_size_ise = bid_size_15; ask_size_ise = ask_size_15; end; else 
	if exchange_id_16 = exchange then do; ask_ise = ask_16; bid_ise = bid_16; bid_size_ise = bid_size_16; ask_size_ise = ask_size_16; end; 

drop exchange exchange_id_1-exchange_id_16 bid_size_1-bid_size_16 ask_size_1-ask_size_16 id;
run;
*/
