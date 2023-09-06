from pycirclize import Circos
import abcd.data.VARS as VARS

def plot_network_connections(connections, values):
    networks = list(VARS.NETWORKS.keys())

    sectors = {n: 1 for n in networks}
    
    magnitudes = {n: 0 for n in networks}
    for connection, value in zip(connections, values):
        n1, n2 = connection.split('-')
        magnitudes[n1] += abs(value)
        magnitudes[n2] += abs(value)
        
    name2color = {'ad': '#F261A8', 'cgc': '#D27DFF', 'ca': '#AB90F4', 'dt': '#EB6065', 'dla': '#A7DF8D', 'fo': '#F6DE62', 'n': '#D4D4D4', 'rspltp': '#F2D3BA', 'sa': '#848484', 'smh': '#92E4E0', 'smm': '#FE8855', 'vta': '#4CB1B0', 'vs': '#5E89F8'}

    circos = Circos(sectors, space=2)
    for sector in circos.sectors:
        sector.text(sector.name+': {:.2f}'.format(magnitudes[sector.name]), size=20)
        track = sector.add_track((95, 100))
        track.axis(fc=name2color[sector.name])
        #track.text(sector.name, color="#171717", size=12)
        track.xticks_by_interval(1)
    
    for connection, value in zip(connections, values):
        n1, n2 = connection.split('-')
        color = "#A2272C" if value > 0 else "#145FB7"
        if n1 == n2:
            circos.link((n1, 0, abs(value)), (n2, 1, 1 - abs(value)), color=color)
        else:
            circos.link((n1, 0, abs(value)), (n2, abs(value), 0), color=color)

    return circos.plotfig()