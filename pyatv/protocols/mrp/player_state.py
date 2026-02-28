"""Module responsible for keeping track of media player states."""

from itertools import chain
import logging
import math
import plistlib
from typing import Any, Dict, List, Optional
import weakref

from pyatv.protocols.mrp import protobuf as pb
from pyatv.protocols.mrp.protocol import MrpProtocol

_LOGGER = logging.getLogger(__name__)


def _audio_format_to_dict(fmt) -> Optional[Dict[str, Any]]:
    """Convert an AudioFormat protobuf message to a plain dict.

    Only includes fields that are actually set on the message.
    Returns None if no fields are set.
    """
    result = {}
    for field in (
        "tier", "bitrate", "sampleRate", "bitDepth", "codec",
        "spatialized", "multiChannel", "channelLayout",
        "audioChannelLayoutDescription",
    ):
        if fmt.HasField(field):
            val = getattr(fmt, field)
            result[field] = int(val) if field == "tier" else val
    return result if result else None


def _audio_route_to_dict(route) -> Optional[Dict[str, Any]]:
    """Convert an AudioRoute protobuf message to a plain dict.

    Only includes fields that are actually set on the message.
    Returns None if no fields are set.
    """
    result = {}
    if route.HasField("type"):
        result["type"] = int(route.type)
    if route.HasField("name"):
        result["name"] = route.name
    if route.HasField("supportsSpatialization"):
        result["supportsSpatialization"] = route.supportsSpatialization
    return result if result else None


def _unarchive_nskeyed(data: bytes) -> Optional[Dict[str, Any]]:
    """Unarchive an NSKeyedArchiver binary plist into a plain dict.

    Apple encodes NSDictionary objects using NSKeyedArchiver format,
    which wraps the actual key-value pairs in $archiver/$objects/$top
    structure. This function unwraps that into a simple Python dict.
    """
    try:
        plist = plistlib.loads(data)
    except Exception:
        return None

    if not isinstance(plist, dict):
        return None

    # If it's already a plain dict (not NSKeyedArchiver), return as-is
    archiver = plist.get("$archiver")
    if archiver != "NSKeyedArchiver":
        return plist

    objects = plist.get("$objects", [])
    top = plist.get("$top", {})
    if not objects or not top:
        return None

    # Get the root object reference
    root_ref = top.get("root")
    if root_ref is None:
        for v in top.values():
            if hasattr(v, "integer"):
                root_ref = v
                break
            elif isinstance(v, int):
                root_ref = v
                break
        if root_ref is None:
            return None

    root_uid = int(root_ref)
    if root_uid >= len(objects):
        return None

    root_obj = objects[root_uid]
    if not isinstance(root_obj, dict):
        return None

    def _resolve_object(obj):
        """Recursively resolve an object from the $objects array."""
        if isinstance(obj, dict):
            # Check if it's an NSDictionary with NS.keys/NS.objects
            ns_keys = obj.get("NS.keys", [])
            ns_values = obj.get("NS.objects", [])
            if ns_keys and ns_values and len(ns_keys) == len(ns_values):
                resolved = {}
                for k_ref, v_ref in zip(ns_keys, ns_values):
                    k_uid = int(k_ref)
                    v_uid = int(v_ref)
                    if k_uid < len(objects) and v_uid < len(objects):
                        key = objects[k_uid]
                        value = objects[v_uid]
                        if isinstance(key, str) and value != "$null":
                            resolved[key] = _resolve_object(value)
                return resolved
            # Check if it's an NSArray with NS.objects
            ns_array = obj.get("NS.objects")
            if ns_array is not None and "NS.keys" not in obj:
                return [
                    _resolve_object(objects[int(ref)])
                    for ref in ns_array
                    if int(ref) < len(objects)
                ]
        return obj

    # Resolve the root NSDictionary
    ns_keys = root_obj.get("NS.keys", [])
    ns_values = root_obj.get("NS.objects", [])

    if not ns_keys or not ns_values or len(ns_keys) != len(ns_values):
        return None

    result = {}
    for k_ref, v_ref in zip(ns_keys, ns_values):
        k_uid = int(k_ref)
        v_uid = int(v_ref)
        if k_uid < len(objects) and v_uid < len(objects):
            key = objects[k_uid]
            value = objects[v_uid]
            if isinstance(key, str) and value != "$null":
                result[key] = _resolve_object(value)

    return result if result else None

DEFAULT_PLAYER_ID = "MediaRemote-DefaultPlayer"


class PlayerState:
    """Represent what is currently playing on a device."""

    def __init__(self, parent, player: pb.NowPlayingPlayer):
        """Initialize a new PlayerState instance."""
        self._playback_state = None
        self.supported_commands: List[pb.CommandInfo] = []
        self.items: List[pb.ContentItem] = []
        self.location: int = 0

        self.identifier: Optional[str] = player.identifier
        self.display_name: Optional[str] = None
        self.parent = parent
        self.update(player)

    @property
    def is_valid(self):
        """Return if player has a valid identifier."""
        return self.identifier is not None and self.identifier != ""

    def update(self, player: pb.NowPlayingPlayer):
        """Update player metadata."""
        self.display_name = player.displayName or self.display_name

    @property
    def playback_state(self):  # pylint: disable=too-many-return-statements
        """Playback state of device."""
        # if playback state has not been received, assume player is not playing
        # anything (i.e. idle)
        if self._playback_state is None:
            return None

        # If player is considered paused, no content is playing
        if self._playback_state == pb.PlaybackState.Paused:
            # ...unless something is in the queue...
            if self.metadata is not None:
                return pb.PlaybackState.Paused
            return None

        # All other states than playing (and paused) should pass through
        if self._playback_state != pb.PlaybackState.Playing:
            return self._playback_state

        playback_rate = self.metadata_field("playbackRate")
        if playback_rate is None:
            return self._playback_state

        if math.isclose(playback_rate, 0.0):
            if self._playback_state == pb.PlaybackState.Playing:
                return pb.PlaybackState.Playing
            return pb.PlaybackState.Paused
        if math.isclose(playback_rate, 1.0):
            return pb.PlaybackState.Playing
        return pb.PlaybackState.Seeking

    @property
    def metadata(self):
        """Metadata of currently playing item."""
        if len(self.items) >= (self.location + 1):
            return self.items[self.location].metadata
        return None

    @property
    def item_identifier(self):
        """Return identifier of current item in queue."""
        if len(self.items) >= (self.location + 1):
            return self.items[self.location].identifier
        return None

    def metadata_field(self, field):
        """Return a specific metadata field or None if missing."""
        metadata = self.metadata
        if metadata and metadata.HasField(field):
            return getattr(metadata, field)
        return None

    def _parsed_plist(self, proto_field: str) -> Optional[Dict[str, Any]]:
        """Parse and cache a binary plist from a protobuf bytes field."""
        cache_key = f"_plist_cache_{proto_field}"
        item_id = self.item_identifier

        # Check cache: (item_identifier, parsed_dict)
        cached = getattr(self, cache_key, None)
        if cached is not None and cached[0] == item_id:
            return cached[1]

        raw = self.metadata_field(proto_field)
        if raw is None:
            result = None
        else:
            result = _unarchive_nskeyed(raw)

        setattr(self, cache_key, (item_id, result))
        return result

    def nowplaying_info_field(self, key: str) -> Any:
        """Return a value from the parsed nowPlayingInfoData plist."""
        parsed = self._parsed_plist("nowPlayingInfoData")
        if parsed:
            return parsed.get(key)
        return None

    def parsed_plist_field(self, proto_field: str, key: str) -> Any:
        """Return a value from any parsed binary plist protobuf field."""
        parsed = self._parsed_plist(proto_field)
        if parsed:
            return parsed.get(key)
        return None

    def raw_plist(self, proto_field: str) -> Optional[Dict[str, Any]]:
        """Return the full parsed plist dict for a protobuf bytes field."""
        return self._parsed_plist(proto_field)

    def command_info(self, command):
        """Return supported command info."""
        for cmd in chain(self.supported_commands, self.parent.supported_commands):
            if cmd.command == command:
                return cmd
        return None

    def handle_set_state(self, setstate):
        """Update current state with new data from SetStateMessage."""
        if setstate.HasField("playbackState"):
            self._playback_state = setstate.playbackState

        if setstate.HasField("supportedCommands"):
            self.supported_commands = setstate.supportedCommands.supportedCommands

        if setstate.HasField("playbackQueue"):
            queue = setstate.playbackQueue
            self.items = queue.contentItems
            self.location = queue.location

    def handle_content_item_update(self, item_update):
        """Update current state with new data from ContentItemUpdate."""
        for updated_item in item_update.contentItems:
            for existing in self.items:
                if updated_item.identifier == existing.identifier:
                    # Other parts of the ContentItem should be merged as
                    # well, but those are not used right now so will do that
                    # when needed.
                    #
                    # NB: MergeFrom will append repeated fields (which is
                    # likely not what is expected)!
                    existing.metadata.MergeFrom(updated_item.metadata)

    def __eq__(self, other):
        """Compare if instance is equal to other instance."""
        return other and self.identifier == other.identifier


class Client:
    """Represent an MRP media player client."""

    def __init__(self, client: pb.NowPlayingClient) -> None:
        """Initialize a new Client instance."""
        self.bundle_identifier: str = client.bundleIdentifier
        self.display_name: Optional[str] = None
        self._active_player: Optional[PlayerState] = None
        self.players: Dict[str, PlayerState] = {}
        self.supported_commands: List[pb.CommandInfo] = []
        self.update(client)

    @property
    def active_player(self) -> PlayerState:
        """Return currently active player."""
        if self._active_player is None:
            if DEFAULT_PLAYER_ID in self.players:
                return self.players[DEFAULT_PLAYER_ID]
            return PlayerState(self, pb.NowPlayingPlayer())
        return self._active_player

    @active_player.setter
    def active_player(self, other: Optional[PlayerState]):
        """Change active player for client."""
        self._active_player = other

    def get_player(self, player: pb.NowPlayingPlayer) -> PlayerState:
        """Get state for a player."""
        if player.identifier not in self.players:
            self.players[player.identifier] = PlayerState(self, player)
        return self.players[player.identifier]

    def handle_set_default_supported_commands(self, supported_commands) -> None:
        """Update default supported commands for client."""
        self.supported_commands = supported_commands.supportedCommands.supportedCommands

    def handle_set_now_playing_player(self, player: pb.NowPlayingPlayer) -> None:
        """Handle change of now playing player."""
        self.active_player = self.get_player(player)

        if self.active_player.is_valid:
            _LOGGER.debug(
                "Active player is now %s (%s)",
                self.active_player.identifier,
                self.active_player.display_name,
            )
        else:
            _LOGGER.debug("Active player no longer set")

    def update(self, client: pb.NowPlayingClient) -> None:
        """Update client metadata."""
        self.display_name = client.displayName or self.display_name


class PlayerStateManager:
    """Manage state of all media players."""

    def __init__(self, protocol: MrpProtocol):
        """Initialize a new PlayerStateManager instance."""
        self.protocol = protocol
        self.volume_controls_available = None
        self._active_client = None
        self._clients: Dict[str, Client] = {}
        self._listener = None
        self._add_listeners()

    def _add_listeners(self) -> None:
        listeners = {
            pb.SET_STATE_MESSAGE: self._handle_set_state,
            pb.UPDATE_CONTENT_ITEM_MESSAGE: self._handle_content_item_update,
            pb.SET_NOW_PLAYING_CLIENT_MESSAGE: self._handle_set_now_playing_client,
            pb.SET_NOW_PLAYING_PLAYER_MESSAGE: self._handle_set_now_playing_player,
            pb.UPDATE_CLIENT_MESSAGE: self._handle_update_client,
            pb.REMOVE_CLIENT_MESSAGE: self._handle_remove_client,
            pb.REMOVE_PLAYER_MESSAGE: self._handle_remove_player,
            pb.SET_DEFAULT_SUPPORTED_COMMANDS_MESSAGE: self._handle_set_default_supported_commands,  # pylint: disable=line-too-long # noqa
        }
        for message, handler in listeners.items():
            self.protocol.listen_to(message, handler)

    def get_client(self, client: pb.NowPlayingClient) -> Client:
        """Return client based on player path."""
        bundle = client.bundleIdentifier
        if bundle not in self._clients:
            self._clients[bundle] = Client(client)
        return self._clients[bundle]

    def get_player(self, player_path: pb.PlayerPath) -> PlayerState:
        """Return player based on a player path."""
        return self.get_client(player_path.client).get_player(player_path.player)

    @property
    def listener(self):
        """Return current listener."""
        if self._listener is None:
            return None
        return self._listener()

    @listener.setter
    def listener(self, new_listener):
        """Change current listener."""
        if new_listener is not None:
            self._listener = weakref.ref(new_listener)
        else:
            self._listener = None

    @property
    def client(self) -> Optional[Client]:
        """Return currently active client."""
        return self._active_client

    @property
    def playing(self) -> PlayerState:
        """Player state for active media player."""
        if self._active_client:
            return self._active_client.active_player
        return PlayerState(Client(pb.NowPlayingClient()), pb.NowPlayingPlayer())

    async def _handle_set_state(self, message):
        setstate = message.inner()

        player = self.get_player(setstate.playerPath)
        player.handle_set_state(setstate)

        await self._state_updated(player=player)

    async def _handle_content_item_update(self, message):
        item_update = message.inner()

        player = self.get_player(item_update.playerPath)
        player.handle_content_item_update(item_update)

        await self._state_updated(player=player)

    async def _handle_set_now_playing_client(self, message):
        self._active_client = self.get_client(message.inner().client)

        _LOGGER.debug("Active client is now %s", self._active_client.bundle_identifier)

        await self._state_updated()

    async def _handle_set_now_playing_player(self, message):
        set_now_playing = message.inner()

        client = self.get_client(set_now_playing.playerPath.client)
        client.handle_set_now_playing_player(set_now_playing.playerPath.player)

        await self._state_updated(client=client)

    async def _handle_remove_client(self, message):
        client_to_remove = message.inner().client

        if client_to_remove.bundleIdentifier in self._clients:
            client = self._clients[client_to_remove.bundleIdentifier]
            del self._clients[client_to_remove.bundleIdentifier]

            if client == self._active_client:
                self._active_client = None
                await self._state_updated()

    async def _handle_remove_player(self, message):
        player_to_remove = message.inner().playerPath

        player = self.get_player(player_to_remove)
        if player.is_valid:
            client = self.get_client(player_to_remove.client)
            del client.players[player.identifier]
            player.parent = None

            if player == client.active_player:
                client.active_player = None
                await self._state_updated(client=client)

    async def _handle_set_default_supported_commands(self, message):
        supported_commands = message.inner()

        client = self.get_client(supported_commands.playerPath.client)
        client.handle_set_default_supported_commands(supported_commands)

        await self._state_updated()

    async def _handle_update_client(self, message):
        update_client = message.inner()

        client = self.get_client(update_client.client)
        client.update(update_client.client)

        await self._state_updated(client=client)

    async def _state_updated(self, client=None, player=None):
        is_active_client = client == self.client
        is_active_player = player == self.playing
        is_always = client is None and player is None

        if is_active_client or is_active_player or is_always:
            if self.listener:
                await self.listener.state_updated()
